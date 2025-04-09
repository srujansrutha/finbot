import streamlit as st
import boto3
import pandas as pd
import ast
import re
from io import BytesIO
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from botocore.exceptions import ClientError
# Load environment variables
load_dotenv()
# AWS Configuration
AWS_CONFIG = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "region_name": os.getenv("AWS_REGION", "us-east-1")
}
# Initialize Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    **AWS_CONFIG
)
# Initialize LLM
llm = BedrockLLM(
    client=bedrock,
    model_id="amazon.titan-text-express-v1",
    model_kwargs={"temperature": 0.3, "maxTokenCount": 4096}
)
# Allowed Services Configuration
SERVICE_CONFIG = {
    "ec2": {"describe_instances": "Reservations[*].Instances[*]"},
    "s3": {"list_buckets": "Buckets[*]"},
    "rds": {"describe_db_instances": "DBInstances[*]"},
    "ebs": {"describe_volumes": "Volumes[?State=='available']"}
}
PROMPT_TEMPLATE = """
Generate Python code to retrieve AWS resource information based on the user request.
Follow these rules strictly:
Allowed Services/Methods:
{services_list}
Code Requirements:
- Use boto3 client
- Handle all errors with try/except blocks
- Store results in 'result' variable
- Return list of dictionaries
- Include only these fields for each service:
{field_examples}
Example for EC2:
```python
try:
    client = boto3.client('ec2')
    response = client.describe_instances()
    result = [
        {{'InstanceId': instance['InstanceId'], 'State': instance['State']['Name'], 'LaunchTime': str(instance['LaunchTime'])}}
        for reservation in response['Reservations']
        for instance in reservation['Instances']
    ]
except Exception as e:
    result = f"Error: {{str(e)}}"
```
User Request: {request}
Generated Code:
"""
def generate_services_list():
    """Generate formatted services list for prompt"""
    return "\n".join(
        f"- {service.upper()}: {', '.join(methods.keys())}"
        for service, methods in SERVICE_CONFIG.items()
    )
def generate_field_examples():
    """Generate field examples for prompt"""
    examples = []
    for service, methods in SERVICE_CONFIG.items():
        for method, jmespath in methods.items():
            examples.append(f"{service}.{method}: Include {jmespath.split('[')[0]} fields")
    return "\n".join(examples)
def validate_code(code):
    """Validate generated code against allowed patterns"""
    service_check = any(
        f"boto3.client('{service}')" in code
        for service in SERVICE_CONFIG
    )
    method_check = any(
        f".{method}(" in code
        for service in SERVICE_CONFIG
        for method in SERVICE_CONFIG[service]
    )
    return service_check and method_check
def sanitize_code(raw_code):
    """Clean and format generated code"""
    code_match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = raw_code.strip()
    code = re.sub(r"#.*", "", code)
    code = "\n".join([line for line in code.split("\n") if line.strip()])
    return code
def execute_safe(code):
    """Execute code with enhanced error handling"""
    exec_env = {"boto3": boto3, "pd": pd, "result": None}
    try:
        ast.parse(code)
        exec(code, exec_env)
        result = exec_env["result"]
        if isinstance(result, str) and result.startswith("Error:"):
            return result
        return result
    except ClientError as e:
        error_code = e.response['Error']['Code']
        msg = f"AWS {error_code} Error: {e.response['Error']['Message']}"
        if error_code == 'AccessDenied':
            msg += "\n\nPlease check your IAM permissions for this service."
        return msg
    except Exception as e:
        return f"Execution Error: {str(e)}"
# Streamlit Interface
st.title("AWS FinOps Agent")
# Initialize session state
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
# Main interface
query = st.text_input("Enter your resource request:", "List all RDS databases")
if st.button("Generate Code"):
    with st.spinner("Generating code..."):
        try:
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = prompt | llm | StrOutputParser()
            generated_code = chain.invoke({
                "request": query,
                "services_list": generate_services_list(),
                "field_examples": generate_field_examples()
            })
            cleaned_code = sanitize_code(generated_code)
            st.session_state.generated_code = cleaned_code
            st.session_state.query_history.insert(0, {
                "timestamp": datetime.now(),
                "query": query,
                "code": cleaned_code,
                "executed": False
            })
        except Exception as e:
            st.error(f"Generation Error: {str(e)}")
if st.session_state.generated_code:
    st.subheader("Generated Code")
    edited_code = st.text_area("Edit code:", value=st.session_state.generated_code, height=300)
    col1, col2 = st.columns([1, 3])
    with col1:
        execute_btn = st.button(":rocket: Execute")
    with col2:
        regenerate_btn = st.button(":arrows_counterclockwise: Regenerate Code")
    if regenerate_btn:
        with st.spinner("Regenerating code..."):
            try:
                prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                chain = prompt | llm | StrOutputParser()
                generated_code = chain.invoke({
                    "request": query,
                    "services_list": generate_services_list(),
                    "field_examples": generate_field_examples()
                })
                cleaned_code = sanitize_code(generated_code)
                st.session_state.generated_code = cleaned_code
                st.rerun()
            except Exception as e:
                st.error(f"Regeneration Error: {str(e)}")
    if execute_btn:
        if validate_code(edited_code):
            with st.spinner("Executing AWS API call..."):
                result = execute_safe(edited_code)
                st.session_state.query_history[0].update({
                    "executed": True,
                    "result": result,
                    "timestamp": datetime.now()
                })
                if isinstance(result, str):
                    if "AccessDenied" in result:
                        st.error(f":closed_lock_with_key: Permission Error\n\n{result}")
                    else:
                        st.error(f":no_entry: Execution Error\n\n{result}")
                elif isinstance(result, list):
                    if len(result) > 0:
                        try:
                            df = pd.DataFrame(result)
                            st.success(f":white_check_mark: Found {len(result)} resources")
                            # Display data
                            st.dataframe(df)
                            # Export options
                            csv = df.to_csv(index=False).encode()
                            st.download_button(
                                ":inbox_tray: Download CSV",
                                data=csv,
                                file_name="aws-resources.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Data conversion error: {str(e)}")
                    else:
                        st.info(":mailbox_with_no_mail: No matching resources found")
                else:
                    st.warning(":warning: Unexpected result format")
        else:
            st.error(":lock: Blocked: Code contains unauthorized operations")
# Query History in Sidebar
st.sidebar.header("Execution History")
if st.session_state.query_history:
    for idx, entry in enumerate(st.session_state.query_history[:5]):
        with st.sidebar.expander(f"{entry['timestamp'].strftime('%H:%M')} - {entry['query']}"):
            st.code(entry['code'])
            if entry['executed']:
                if isinstance(entry['result'], list):
                    try:
                        df = pd.DataFrame(entry['result'])
                        st.write(f"Results: {len(df)} rows")
                        st.dataframe(df.head(3))
                        st.download_button(
                            f":inbox_tray: Download {idx}",
                            data=df.to_csv(index=False).encode(),
                            file_name=f"history-{idx}.csv",
                            mime="text/csv"
                        )
                    except:
                        st.write("Could not display results")
                else:
                    st.write(entry['result'])
st.markdown("---")
st.caption(f"Supported Services: {', '.join(SERVICE_CONFIG.keys())}")