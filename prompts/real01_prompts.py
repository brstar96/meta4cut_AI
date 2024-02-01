real01_prompts = dict({
    "v2023.03.01" : {
        # examples from https://github.com/wibus-wee/stable_diffusion_chilloutmix_ipynb/blob/main/prompts.md#sfw-cat-ears--blue-eyes
        "positive": { 
            'basic' : "korea street, tenis skirt, jeans,", 
            'chilloutmix' : 'girl standing in cozy room, 1girl, ultra highres, small head, aegyo sal, puffy eyes, small breasts, ulzzang, beautiful, cute korean girl, korean beauty, high quality, photorealistic, sharp focus, subsurface scattering', 
            'RealDosMix' : 'photo studio, 1girl, ultra highres, aegyo sal, puffy eyes, small breasts, ulzzang, beautiful, cute korean girl, korean beauty, high quality, photorealistic, sharp focus, subsurface scattering', 
            'henmixReal' : 'studio, street, 1girl, ultra highres, small head, aegyo sal, puffy eyes, small breasts, ulzzang, beautiful, cute korean girl, korean beauty, high quality, photorealistic, sharp focus, subsurface scattering',
            # 'Cat ears + Blue eyes':'best quality, ultra high res, (photorealistic:1.4), 1 white child, (ulzzang-6500:1.0), smiling, (PureErosFace_V1:1.0), ((detailed facial features)), alluring blue eyes, F/2.8, HDR, 8k resolution, ulzzang-6500, (kpop idol), aegyo sal, from side, looking at camera, cat ears, sports bra', 
            # 'School uniform':'best quality, ultra high res, (photorealistic:1.4), 1girl,school uniform,cute,(platinum blonde grey hair:1), ((puffy eyes)), looking at viewer, full body', 
            # 'White sports bra + Outdoors':'best quality, ultra high res, (photorealistic:1.4), 1girl, loose and oversized black jacket, white sports bra, (green yoga pants:1), (Kpop idol), (aegyo sal:1), (light brown short ponytail:1.2), ((puffy eyes)), looking at viewer, smiling, cute, full body, streets, outdoors', 
            # 'Platinum blonde hair + Black skirt':'best quality, ultra high res, (photorealistic:1.4), 1woman, sleeveless white button shirt, black skirt, black choker, cute, (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, full body, facing front, masterpiece, best quality,', 
            }, 
        "negative" : {
            'basic' : 'nsfw, sexy, extra face, revealing, panty, underwear, lingerie, exposing, hot, loli, child, swimsuit, monokini, bloomer, latex, bodysuit, pompom, hat, cap, prop on hand, extra arms, extra hands, extra legs, extra head, long neck, extra fingers, fused fingers, blurry, paintings', 
            'chilloutmix' : 'hat, grab on hand, fantasy, extra face, flutter, nsfw, sexy, exposing, loli, child, highleg, panty, underwear, swimsuit, monokini, paintings, sketches, worst quality, low quality, lowres, monochrome, bad feet, ((wrong feet)), (wrong shoes), bad hands, distorted, missing fingers, multiple feet, bad knees, extra fingers, extra arms, bad proportions, nipples, watermark, signature, text', 
            'RealDosMix' : 'fantasy, extra face, extra eyes, flutter, nsfw, sexy, exposing, loli, child, sketches, worst quality, low quality, lowres, monochrome, bad feet, ((wrong feet)), (wrong shoes), distorted, missing fingers, multiple feet, bad knees, extra fingers, extra arms, bad proportions, nipples, watermark, signature, text, background',
            'henmixReal' :'hat, grab on hand, fantasy, extra face, flutter, nsfw, sexy, exposing, loli, child, highleg, panty, underwear, swimsuit, monokini, paintings, sketches, worst quality, low quality, lowres, monochrome, bad feet, ((wrong feet)), (wrong shoes), bad hands, distorted, missing fingers, multiple feet, bad knees, extra fingers, extra arms, bad proportions, nipples, watermark, signature, text',
            # 'Cat ears + Blue eyes':'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans', 
            # 'School uniform':'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw,', 
            # 'White sports bra + Outdoors':'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw', 
            # 'Platinum blonde hair + Black skirt':'paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry', 
            },

        "positive_gpt_prompt_v0.1": 
            'I am going to draw a realistic photo that covers the above sentences. \n \
            Please describe in detail the korean instagram girl who will appear in the illustration without pose and NSFW expressions. A simple 2~4 word prompt format that distinguishes features with commas. \
            This will be entered in the drawing ai called stable diffusion. There should be a description of the costume, hair style and color, and filming information. Here is an example. \n\n \
            "twintale, ponytale, Kpop idol, top quality, movie lighting, depth of field, lens flare" \n\n\
            First, create 5 prompts for the "Hair Style and Color" categories without category name. \
            Second, create 5 prompts for the "Chatacter Cloth and Emotion" categories without category name. \
            Last, create 5 prompts for the "Filming Information" categories without category name. \n\n\
            Then merge the three lists sequentially and return them in the form of a "python list" which length is 5. Do not add numbers, just list them.', 
        "positive_gpt_prompt_v0.2": {
            'q_part1': # guide for the first part of the prompt
                '# Instructions \n \
                You are now asked to create a prompt for the image generation Al, Stable Diffusion. Referring to the "Example", please issue a prompt that generates a "Generate Target" based on the following "Properties of Prompts". \n\n \
                # Properties of Prompts \n \
                - There are "positive prompts".\n \
                - An image is generated that contains certain elements that are positive prompts.\n \
                - Prompts can use up to 75 tokens. The counting of tokens is the same as the tokeniser in the large language model.\n \
                - Prompts are tokenised and interpreted in order from the beginning, just like the input part of the large language model.\n \
                - The words at the beginning are more sensitive.\n \
                - The influence of the token can be increased by a factor of 1.1 by enclosing it in brackets O. (token:1.5) would increase the influence of the token by a factor of 1.5.\n \
                - Tokens in close proximity are interpreted by association. In particular, three consecutive tokens are strongly associated.\n \
                - For example, the inclusion of "black hair girl brown eyes" in a prompt is likely to produce a girl with black hair, while the sequence of "hair girl brown" tokens is also associated, so a girl with brown hair may be output.\n \
                - Words such as "best quality" or "ultra highres" can be used to improve the quality of the image. Conversely, including words such as "worst quality" in a negative prompt can also improve quality.\n \
                - A prompt that is too long will produce poor results.\n \
                # Example\n \
                ## 15 year old girl with long blonde lightly wavy hair and blue eyes\n \
                Positive prompt: photorealistic, movie lighting, depth of field, lens flare, 15yo, (wavy hair:0.9), blue eyes, flat chest, face close up \n\n\
                I am going to draw a realistic girl photo. First, please refer to the rules above to create an 2~4 word answer especially about "Filming Information".\n\n', 
            'q_part2':
                'Second, Read the sentence above and select the only one clothes and only one situations that appeared the most in the form of a Python list.\n \
                Then merge the First and Second lists you generated sequentially and return them in the form of a "python list". Do not add numbers or category name, just list words.'
                }, 
        "positive_gpt_prompt_v0.2.1": 'Read the sentence above and select the top & bottom clothes, only one situations and emotions of a woman that appeared the most in the form of a "Python list". Do not add numbers, just list them.', 

        "lora": {
            "ulzzang-6500-v1.1" : {
                "positive_prompt":'1girl, (ulzzang-6500-v1.1:0.5), (shiny skin:0.7), (korean beauty, k-pop idol makeups:0.9), (wide hips:1.1), (thigh gap:1.0), perfect face, blush, perfect body, pretty face', 
                "negative_prompt":'(Nsfw, brown, Nipples, pussy, Chinese, pubic hair, penis, labia minora:1.3), (big eyes, big head:0.6),\
                                  (Low Quality, worst quality, Low-res:1.4), (mascara:0.3)'
                }, 
            "koreanDollLikeness_v15" : {
                "positive_prompt":'<lora:koreanDollLikeness_v15:1.3>, ultra detailed, highres, (realistic, photo-realistic:1.4), 8k, \
                                  1girl, distortion, (best quality), physically-based rendering, korean beauty, instagram, perfect face, blush, perfect body, pretty face, depth of field, solo, F2.4, 35mm', 
                "negative_prompt":'ng_deepnegative_v1_75t, paintings, sketches, (low quality:2), (normal quality:2), (worst quality:2), \
                                  lowres, ((monochrome)), ((grayscale)), acnes, skin spots, age spot, skin blemishes, \
                                  bad feet, ((wrong feet)), (wrong shoes), bad hands, distorted, missing fingers, multiple feet, bad knees, extra fingers, extra arms'
                }
            }, 

        "img_caption_exp": 
            'a woman in jeans and white shirt is dancing \
            a woman in jeans and a white shirt is dancing \
            a woman in jeans and a white shirt is standing on a bed \
            a woman in jeans and white shirt standing on a bed'
    }    
})

'''
* Model Parameter Suggestions
 - steps: 40
 - sampler: DPM++ SDE Karras
 - seed: 1862705621
 - resolution: 576x768

'''