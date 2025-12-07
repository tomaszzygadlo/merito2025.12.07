import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_KEY"])

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Jesteś tajnym agentem jej królewskiej mości z licencją na zabijanie. Nazywasz się James Bond."},
        {"role": "user", "content": "Jak się nazywasz?"}
    ]
)

print(response.choices[0].message.content)
