from openai import OpenAI

client = OpenAI(
    api_key="7b7e3cbc-f103-510b-8eea-c5176c30f420",
    base_url="https://arvancloudai.ir/gateway/models/GPT-5.4-Nano/Vq1xDrIePANZZtlgnxW0-Fz_g9tg5hVc4_eajZz7aiRIT2gfuQONqwMBxBuoXNB0_z-GMeHNFJFiPWKhzRmO9a0Rb_X-EKhqZH3VRHLz2Y_ogiHWUHy4lhZk-hFkma6-ZeRcNW6GRVBNski8uv5Su4aP9Qq7ExFfiHL9pzH-seqybFSfFAGCdILolfNtX5odNxhpF3gQBCWV2SekC3n1yIEdoB2pG3uJoL8O5Gl8hVOKJByzVDVYNA/v1"
)

response = client.chat.completions.create(
    model="GPT-5.4-Nano",
    messages=[
        {
            "role": "user",
            "content": "who is the president of IRAN!"
        }
    ]
)

print(response.choices[0].message.content)