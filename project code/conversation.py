from config import SYSTEM_PROMPT
def get_system_prompt() :
    return SYSTEM_PROMPT
def build_convo(input_type, question, input_data, extra_content=None):
    content_item = {
        "type": input_type,
        input_type: input_data,
    }
    if extra_content:
        content_item.update(extra_content)

    message = [
        {
            "role":"system",
            "content": [ {
                "type":"text",
                "text": get_system_prompt()
            }
            ]
            
        },{
            "role":"user",
            "content": [
                content_item,
                {
                    "type":"text",
                    "text":question
                }
            ]
        }
    ]
    return message