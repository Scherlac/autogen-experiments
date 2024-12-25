
import os  
import base64
from openai import AzureOpenAI  

# Load the environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()


azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL")
model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_version=os.getenv("AZURE_OPENAI_API_VERSION")



# Initialize Azure OpenAI client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=azure_endpoint,  
    api_key=api_key,
    api_version=api_version
)
    
    
# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

#Prepare the chat prompt 
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
            }
        ]
    }
] 
    
# Include speech result if speech is enabled  
messages = chat_prompt  
    
# Generate the completion  
completion = client.chat.completions.create(  
    model=model, 
    messages=messages,  
    max_tokens=800,  
    temperature=0.7,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,  
    stop=None,  
    stream=False
)

print(completion.to_json())  
    