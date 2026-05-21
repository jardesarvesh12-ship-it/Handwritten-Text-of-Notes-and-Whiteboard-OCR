import os
import sys
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError

# Load environment variables from .env file
# This matches how the main application loads settings
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def test_credentials():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("[ERROR] AWS credentials were not found in the .env file.")
        print("Please check that AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in handwritten-ocr/.env")
        return

    print("--- AWS Textract Verification Script ---")
    print(f"Loaded Access Key ID: {AWS_ACCESS_KEY_ID[:8]}... (starts with)")
    print(f"Region: {AWS_REGION}")
    print("Initializing AWS Textract client...")
    
    try:
        client = boto3.client(
            "textract",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

        print("Sending authenticating test request to AWS Textract (DetectDocumentText)...")
        
        # We pass dummy data. A working credential will authenticate successfully 
        # and then return a validation or document format error.
        dummy_document = {
            'Bytes': b'dummy data'
        }

        response = client.detect_document_text(Document=dummy_document)
        print("[SUCCESS] Connection and credentials are fully working!")
        print(response)
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', '')
        
        print("\n--- AWS Response Result ---")
        print(f"Error Code: {error_code}")
        print(f"Error Message: {error_message}")
        
        if error_code in ["UnrecognizedClientException", "InvalidClientTokenId", "SignatureDoesNotMatch", "AuthFailure"]:
            print("\n[CRITICAL ERROR] The AWS credentials are INVALID. Authentication failed.")
        elif error_code == "AccessDeniedException":
            print("\n[CRITICAL ERROR] The AWS credentials are VALID, but they do NOT have IAM permission/policies to access Textract.")
        elif error_code in ["UnsupportedDocumentException", "ValidationException", "InvalidParameterException"]:
            print("\n[SUCCESS] The credentials are VALID and WORKING!")
            print("The AWS server successfully authenticated and authorized your request.")
            print("It rejected the dummy input as expected, meaning it is ready for real images!")
        else:
            print(f"\n[WARNING] Unexpected AWS ClientError code '{error_code}'. This might mean credentials are valid, but check the error details.")
            
    except NoCredentialsError:
        print("\n[ERROR] No credentials found.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    test_credentials()
