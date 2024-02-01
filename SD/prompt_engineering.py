import openai
import json
import re

def sanitize_prompt(prompt_txt):
    result = []
    for match in re.findall(r"\((.*?)\)|([^,]+)", prompt_txt):
        result.append(match[0] or match[1])

    return ', '.join(result).replace('                                    ', ' ')

def get_gpt_text(gpt_prompt, temp=0.5, max_tokens=256, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    response = openai.Completion.create(engine="text-davinci-003", # text-davinci-003, gpt-3.5-turbo
                                        prompt=gpt_prompt,
                                        temperature=temp,
                                        max_tokens=max_tokens,
                                        top_p=top_p,
                                        frequency_penalty=frequency_penalty,
                                        presence_penalty=presence_penalty
                                        )
    return response['choices'][0]['text']
    
def set_anime_prompts(image_caption, basic_prompt):
    gpt_positive_prompt = '"{}" \n\n {}'.format(image_caption, basic_prompt['positive_gpt_prompt'])
    positive_prompt = basic_prompt['positive']
    negative_prompt = basic_prompt['negative']
    gpt_positive_response = get_gpt_text(gpt_positive_prompt).replace('\n', '').replace('[', '').replace(']', '').replace('+', '').replace("'", '').replace('"', '').replace('.', '')
    pattern = re.compile(r"Hair Style and Color|Character Cloth and Emotion|Filming|Information|Answer:|Bottom:|Top:|;|/|!|~|`|(|)|{|}|<|>|@|#|%|&|=|…|·|•|‧|∙|·|●|○|◎|◇|◆|□|■|△|▲|※|〒|→|←|↑|↓|↔|↕|↖|↗|↘|↙|↩|↪|⇒|⇔|⇒|⇔|") 
    gpt_positive_response = re.sub(pattern, '', gpt_positive_response) 
    gpt_positive_response = gpt_positive_response + positive_prompt
    print('\n*Input Image Caption:\n{}\n\n*Final Positive Response:\n{}\n\n*Final Negative Response:\n{}\n\n'.format(image_caption, gpt_positive_response, negative_prompt))

    return {'positive':gpt_positive_response.replace('\r\n', ''), 'negative':negative_prompt.replace('\r\n', '')}

def set_real01_prompts(image_caption, basic_prompt, model_name='henmixReal', lora_name='koreanDollLikeness_v15', pos_gpt_prompt_version='positive_gpt_prompt_v0.2.1', visual_gpt=True):
    if visual_gpt:
        image_caption = ' '.join(image_caption).replace(' ', '')
        if pos_gpt_prompt_version == 'positive_gpt_prompt_v0.2':
            positive_prompt = basic_prompt['lora']['{}'.format(lora_name)]['positive_prompt'] + ', ' + basic_prompt['positive']['{}'.format(model_name)]
            negative_prompt = basic_prompt['negative']['{}'.format(model_name)] + ', ' + basic_prompt['lora']['{}'.format(lora_name)]['negative_prompt']
            gpt_prompt = '{}'.format(basic_prompt['{}'.fromat(pos_gpt_prompt_version)]['q_part1']+', '+image_caption+', '+basic_prompt['{}'.format(pos_gpt_prompt_version)]['q_part2'])
        elif pos_gpt_prompt_version == 'positive_gpt_prompt_v0.2.1':
            positive_prompt = basic_prompt['positive']['{}'.format(model_name)]
            negative_prompt = basic_prompt['negative']['{}'.format(model_name)]
            gpt_prompt = '{} \n\n {}'.format(image_caption, basic_prompt['{}'.format(pos_gpt_prompt_version)])
    
        gpt_positive_response = get_gpt_text(gpt_prompt).replace('\n', '').replace('[', '').replace(']', '').replace('+', '').replace("'", '').replace('"', '').replace('.', '')
        pattern = re.compile(r"Hair Style and Color|Character Cloth and Emotion|Filming|Information|Positive|positive|Negative|negative|Prompt|prompt|face close up|Answer|Bottom|Top|Clothes:|Situation:|Situation|Emotion|s:|;|/|!|~|`|(|)|{|}|<|>|@|#|%|&|=|…|·|•|‧|∙|·|●|○|◎|◇|◆|□|■|△|▲|※|〒|→|←|↑|↓|↔|↕|↖|↗|↘|↙|↩|↪|⇒|⇔|⇒|⇔| ,|") 
        gpt_positive_response = re.sub(pattern, '', gpt_positive_response).strip()
        positive_prompt = gpt_positive_response + ',' + positive_prompt
        positive_prompt = positive_prompt.replace('                                    ', ' ')
        negative_prompt = negative_prompt.replace('                                    ', ' ')
        print('\n*Input Image Caption:\n{}\n\n*Final Positive Response:\n{}\n\n*Final Negative Response:\n{}\n\n'.format(image_caption, positive_prompt, negative_prompt))
    else:
        positive_prompt = basic_prompt['positive']['{}'.format(model_name)]
        negative_prompt = basic_prompt['negative']['{}'.format(model_name)]
        print('\n*Input Image Caption:\n{}\n\n*Final Positive Response:\n{}\n\n*Final Negative Response:\n{}\n\n'.format(image_caption, positive_prompt, negative_prompt))

    return {'positive':positive_prompt, 'negative':negative_prompt}


def gen_prompt(image_caption, style, basic_prompt, model_name='RealDosMix', lora_name = 'koreanDollLikeness_v15', visual_gpt=True, openai_api_key=None):
    openai.api_key = openai_api_key
    print('*Style: {}'.format(style))

    if style == 'anime' and image_caption is not None:
        prompts = set_anime_prompts(image_caption, basic_prompt)
    elif style == 'real_01' and image_caption == None:
        prompts = set_real01_prompts(image_caption=image_caption, basic_prompt=basic_prompt, model_name=model_name, lora_name=lora_name, visual_gpt=visual_gpt)
    
    return prompts
    
    
if __name__ == "__main__":
    OPENAI_API_KEY = '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'
    style = 'real_01'
    prompt_version = "v2023.03.01"

    if style == 'anime':
        from prompts.anime01_prompts import animstyle_prompts
        base_prompts = animstyle_prompts['{}'.format(prompt_version)]
        gen_prompt(base_prompts['img_caption_exp'], style, base_prompts, OPENAI_API_KEY)
    elif style == 'real_01':
        from prompts.real01_prompts import real01_prompts
        base_prompts = real01_prompts['{}'.format(prompt_version)]
        gen_prompt(base_prompts['img_caption_exp'], style, base_prompts, lora_name = 'koreanDollLikeness_v15', openai_api_key=OPENAI_API_KEY)
