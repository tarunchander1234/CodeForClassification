from huggingface_hub import InferenceClient
import os
import json
import re

os.environ["HF_TOKEN"]="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"

client = InferenceClient(
      api_key="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",
      provider="nebius",
)







completion = client.chat.completions.create(
    model="microsoft/phi-4",
    messages=[
            {"role": "system",
            "content": '''You would be provided with research paper having Id and Abstract. You would need to identify the disease mentioned in abstract. The response should be returned in below format as an example:"
                {
                "abstract_id": "12345",
                "extracted_diseases": ["Lung Cancer", "Breast Cancer"]
                }'''},
            {"role": "user",
            "content": '''ID:30872385
                        Abstract: AIMS: BRAF V600E detection assists in the diagnosis of hairy cell leukaemia (HCL); however, testing practices vary. We evaluated the clinical utility of 5 BRAF mutation testing strategies for use on bone marrow trephines (BMT). METHODS: 11 HCL, 5 HCL 'mimic', 2 treated HCL and 10 normal BMT specimens were tested for mutant BRAF, comparing Sanger sequencing, pyrosequencing, amplicon-based next generation sequencing (NGS), automated (Idylla) PCR and immunohistochemistry (IHC). RESULTS: PCR and IHC were cheaper and identified V600E in 100 % of HCL cases. Pyrosequencing detected the mutation in 91%, NGS in 55% of cases and Sanger sequencing in 27%. All assays gave wild-type BRAF results in HCL mimics and normal BMT samples. CONCLUSIONS: PCR and IHC were most sensitive and cost-effective, but these have limited scope for multiplexing and are likely to be replaced by NGS gene panels or whole genome sequencing in the medium to long term.'''
            }
    ],
    #provider="nebius",
    max_tokens=512,
)

print(completion.choices[0].message["content"])

response_content = completion.choices[0].message["content"]

cleaned_string = re.sub(r"\`\`\`json\s*", "", response_content.strip(), flags=re.IGNORECASE)
cleaned_string = re.sub(r"\`\`\`", "", cleaned_string.strip(), flags=re.IGNORECASE)

print("Cleaned String:", cleaned_string)
# Parse the cleaned string to a Python dictionary
try:
    parsed_json = json.loads(cleaned_string)
    
    # Access the fields
    abstract_id = parsed_json.get("abstract_id")
    diseases = parsed_json.get("extracted_diseases", [])
    
    print("Abstract ID:", abstract_id)
    print("Extracted Diseases:", diseases)

except json.JSONDecodeError as e:
    print("Invalid JSON format:", e)

# response_dict = json.loads(response_content)

# # Extract the values of abstract_id and extracted_diseases
# abstract_id = response_dict.get("abstract_id", "N/A")  # Default to "N/A" if key is missing
# extracted_diseases = response_dict.get("extracted_diseases", [])

# # Print the extracted values
# print(f"Abstract ID: {abstract_id}")
# print(f"Extracted Diseases: {', '.join(extracted_diseases)}")

# import requests

# API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
# headers = {"Authorization": "Bearer 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"}

# os.environ['API_URL']="https://router.huggingface.co/nebius/v1/chat/completions"
# #os.environ['headers']={"Authorization": "Bearer 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"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

# response = query({
#     "messages": [
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ],
#     "max_tokens": 512,
#     "model": "microsoft/phi-4"
# })

# print(response)