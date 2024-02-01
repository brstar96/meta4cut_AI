animstyle_prompts = dict({
    "v2023.03.01" : {
        "positive": 
            "high contrast, masterpiece, ultra-highres, 1girl, korean, anime, japanese, small breasts, \
            best quality, looking at viewer, cute, twintale",

        "negative": 
            "magic, fantasy, flash, nsfw, sexy, revealing, exposing, hot, loli, child, ugly, grey tone, achromatic, \
            skirt, hat, umbrella, parasol, swimsuit, monokini, latex, text, bad anatomy, lingerie, bodysuit, highleg, priest,\
            panty, bloomer, short pants, short jean, underwear, extra arms, extra hands, extra legs, extra head,\
            long neck, shortcut, username, watermark, extra fingers, fused fingers, blurry, paintings, \
            sketches, worst quality, low quality, lowres, monochrome, grain", 

        "positive_gpt_prompt": 
            'I am going to draw a stunning anime style illustration that covers the above sentences. \n \
            Please describe in detail the cute girl who will appear in the illustration without NSFW expressions. A simple 2~4 word prompt format that distinguishes features with commas. \
            This will be entered in the drawing ai called stable diffusion. There should be a description of the costume, hair style and color, pose, and filming information. Here is an example. \n\n \
            "ulzzang-6500, Kpop idol, top quality, canon 5d mark, movie lighting, depth of field, lens flare, looking at viewer" \n\n\
            First, create 5 prompts for the "Hair Style and Color" categories without category name. \
            Second, create 5 prompts for the "Chatacter Cloth and Emotion" categories without category name. \
            Last, create 5 prompts for the "Filming Information" categories without category name. \n\n\
            Then merge the three lists sequentially and return them in the form of a "python list". Do not add numbers, just list them.', 

        "embedding": {
            "ulzzang-6500-v1.1" : {
                "positive_promp":'(ulzzang-6500-v1.1:0.9), (shiny skin:1.4), (korean beauty, k-pop idol makeups:1.3), \
                                  perfect face, blush, perfect body, pretty face, (1girl, 20 year old:1.3)', 
                "negative_promp":'(Nsfw, brown, Nipples, pussy, Chinese, pubic hair, penis, labia minora:1.3), (big eyes, big head:0.6),\
                                  (Low Quality, worst quality, Low-res:1.4), (mascara:0.3)'
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