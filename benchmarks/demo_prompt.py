text_english = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

text_chinese = [
    "请介绍北京的旅游景点",
    "介绍一下大熊猫",
    "晚上睡不着应该怎么办",
    "李白的代表作有哪些？",
]

chat = [
    [
        {
            "role": "system",
            "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        },
        {
            "role": "human",
            "content": "请给出10个要到北京旅游的理由。",
        },
    ],
    [
        {
            "role": "system",
            "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        },
        {
            "role": "human",
            "content": "写一首中秋主题的五言绝句",
        },
    ],
    [
        {
            "role": "system",
            "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        },
        {
            "role": "human",
            "content": "请给出北京最著名的10个旅游景点。",
        },
        {
            "role": "assistant",
            "content": "北京最著名的10个旅游景点包括：长城、故宫、颐和园、天坛、北海公园、恭王府、圆明园、香山公园、中国科学技术馆、北京动物园。",
        },
        {
            "role": "human",
            "content": "一天时间能完成几个景点的游玩？",
        },
    ],
]

deepseek_chat = [
    [
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
    [
        {
            "role": "user",
            "content": "Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference."
        }
    ],
    [
        {
            "role": "user",
            "content": "Write a piece of quicksort code in C++."
        }
    ]
]

character_chat = [
    [
        {
            "user_info": "女儿国国王",
            "bot_info": "孙悟空由开天辟地以来的仙石孕育而生，是一个男性， …，尊为“美猴王”。 …，学会地煞数七十二变…。孙悟空喜欢说“俺老孙”，“俺”来称呼自己。",
            "user_name": "国王",
            "bot_name": "孙悟空",
            "history": [
                ["来者何人？", "俺老孙是五百年前大闹天宫的齐天大圣！"],
                ["听说过，久仰大名啊，你非常厉害呢", "虽然你没什么名气，但俺老孙还喜欢和你相处，嘿嘿嘿！"]
            ],
            "greeting": "谢谢，你能跟我说说你们西天取经的故事吗？"
        },
    ],
    [
        {
            "user_info": "",
            "bot_info": "鲁迅，原名周樟寿，后改名周树人，字豫才，中国知名近代作家，代表作品包括《呐喊》 ....他的性格看得通透，被誉为先生、迅哥儿等，其文体特征以“鲁迅体”闻名。",
            "user_name": "记者",
            "bot_name": "鲁迅",
            "history": [],
            "greeting": "我受刘半农之托，想问你是否愿意被推荐为诺贝尔文学奖的候选人？"
        },
    ],
    [
        {
            "user_info": "我是陆星辰，是一个男性，是一位知名导演，也是苏梦远的合作导演。我擅长拍摄音乐题材的电影。苏梦远对我的态度是尊敬的，并视我为良师益友。",
            "bot_info": "苏梦远，本名苏远心，是一位当红的国内女歌手及演员。在参加选秀节目后，凭借独特的嗓音及出众的舞台魅力迅速成名，进入娱乐圈。她外表美丽动人，但真正的魅力在于她的才华和勤奋。苏梦远是音乐学院毕业的优秀生，善于创作，拥有多首热门原创歌曲。除了音乐方面的成就，她还热衷于慈善事业，积极参加公益活动，用实际行动传递正能量。在工作中，她对待工作非常敬业，拍戏时总是全身心投入角色，赢得了业内人士的赞誉和粉丝的喜爱。虽然在娱乐圈，但她始终保持低调、谦逊的态度，深得同行尊重。在表达时，苏梦远喜欢使用“我们”和“一起”，强调团队精神。",
            "bot_name": "苏梦远",
            "user_name": "陆星辰",
            "history": [],
            "greeting" : "你好"
        },
    ],
    [
        {
            "user_info": "将自己视做男明星女友一般的粉丝群体。",
            "bot_info": "宋亚轩，2004年3月4日出生于山东省滨州市博兴县，中国内地男歌手、演员，男子演唱组合时代少年团成员，目就读于中央戏剧学院。宋亚轩是一个温柔慢热的人，对待他人温柔礼貌，宋亚轩喜欢安静地坐在角落里，不时地陷入自己的“小世界”里开始放空，偶尔还有点蒙，但是这样的宋亚轩显得更可爱了。宋亚轩音乐功底深厚，是一位在不唱歌时惜字如金的“静默”美少年。宋亚轩的口头禅是“真秀”，一般出现在他表示很震惊的时候。",
            "user_name": "女友粉",
            "bot_name": "宋亚轩",
            "history": [],
            "greeting": "把玩偶还给我!!!"
        },
    ],
]

code_completion = [
    "Write a python function to generate the nth fibonacci number.",
    "import argparse\n\ndef main(string: str):\n    print(string)\n    print(string[::-1])\n\nif __name__ == \"__main__\":",
    "\n\ndef string_sequence(n: int) -> str:\n\t\"\"\" Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n\t>>> string_sequence(0)\n\t'0'\n\t>>> string_sequence(5)\n\t'0 1 2 3 4 5'\n\t\"\"\"\n",
]

code_infilling = [
    [
        {
            "role": "user",
            "content": "def remove_non_ascii(s: str) -> str:\n   \"\"\" <FILL_ME>\n   return result",
        },
    ],
    [
        {
            "role": "user",
            "content": "def print_hello_world():\n    <FILL_ME>\n    print('Hello world!')",
        },
    ],
    [
        {
            "role": "user",
            "content": " def alternating(list1, list2):\n   results = []\n   for i in range(min(len(list1), len(list2))):\n       results.append(list1[i])\n       results.append(list2[i])\n   if len(list1) > len(list2):\n       <FILL_ME>\n   else:\n       results.extend(list2[i+1:])\n   return results",
        },
    ],
]

code_instruction = [
    [
        {
            "role": "user",
            "content": "Write a quick sort function in C++",
            "language": "C++",
        },
    ],
    [
        {
            "role": "user",
            "content": "write a bubble sort function in Python",
            "language": "Python",
        },
    ],
    [
        {
            "role": "user",
            "content": "Write a binary tree class in Java",
            "language": "Java",
        },
    ],
]

# test jinja
# import jinja2
# from jinja2 import Environment, FileSystemLoader
# 
# env = Environment(loader=FileSystemLoader("benchmarks/templates"))
# template = env.get_template('template_starcoder2_completion.jinja')
# rendered_output = template.render(messages=code_completion)

# test tokenizer apply chat template
# from vllm.transformers_utils.tokenizer import get_tokenizer
# tokenizer = get_tokenizer("CodeLlama-13b-Instruct-hf", trust_remote_code=True)
#
# with open("default.jinja", "r") as f:
#     template = f.read()
# tokenizer.chat_template = template

# print(tokenizer.apply_chat_template(conversation=code_instruct, tokenize=False))
