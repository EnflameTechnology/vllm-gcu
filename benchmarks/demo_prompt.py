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


code_merge = [
[
		{
			"role": "system",
			"content": "You are a coding assistant, I will give you a source code file and some updated code snippets, help to merge the code snippets into the source code file, get the merged code. \n1. Maintain the structure, order, comments, and indentation of the code. \n2. Please output the complete merged code.\n3. Just output the code, no instructions required.\n\nExample 1: \n \n## Original code file content\n\n```python\ndef hello(): \n    print(\"Hello\") \n```\n \n## Update code snippets\n\n```python\n//... existing code ... \n    print(\"Hello World\") \n```\n \n## The output\n\n```python\ndef hello(): \n    print(\"Hello World\") \n```"
		},
		{
			"role": "user",
			"content": "## Original code file content\n\nfilepath: testdata/speed/30ko.go\n```go\npackage storage\n\nimport (\n\t\"bytes\"\n\t\"fmt\"\n\t\"regexp\"\n\t\"regexp/syntax\"\n\t\"sort\"\n\t\"strings\"\n\t\"sync\"\n\t\"sync/atomic\"\n\n\t\"github.com/VictoriaMetrics/VictoriaMetrics/lib/encoding\"\n\t\"github.com/VictoriaMetrics/VictoriaMetrics/lib/logger\"\n\t\"github.com/VictoriaMetrics/VictoriaMetrics/lib/memory\"\n)\n\n// convertToCompositeTagFilterss converts tfss to composite filters.\n//\n// This converts `foo{bar=\"baz\",x=~\"a.+\"}` to `{foo=bar=\"baz\",foo=x=~\"a.+\"} filter.\nfunc convertToCompositeTagFilterss(tfss []*TagFilters) []*TagFilters {\n\ttfssNew := make([]*TagFilters, 0, len(tfss))\n\tfor _, tfs := range tfss {\n\t\ttfssNew = append(tfssNew, convertToCompositeTagFilters(tfs)...)\n\t}\n\treturn tfssNew\n}\n\nfunc convertToCompositeTagFilters(tfs *TagFilters) []*TagFilters {\n\tvar tfssCompiled []*TagFilters\n\t// Search for filters on metric name, which will be used for creating composite filters.\n\tvar names [][]byte\n\tnamePrefix := \"\"\n\thasPositiveFilter := false\n\tfor _, tf := range tfs.tfs {\n\t\tif len(tf.key) == 0 {\n\t\t\tif !tf.isNegative && !tf.isRegexp {\n\t\t\t\tnames = [][]byte{tf.value}\n\t\t\t} else if !tf.isNegative && tf.isRegexp && len(tf.orSuffixes) > 0 {\n\t\t\t\t// Split the filter {__name__=~\"name1|...|nameN\", other_filters}\n\t\t\t\t// into name1{other_filters}, ..., nameN{other_filters}\n\t\t\t\t// and generate composite filters for each of them.\n\t\t\t\tnames = names[:0] // override the previous filters on metric name\n\t\t\t\tfor _, orSuffix := range tf.orSuffixes {\n\t\t\t\t\tnames = append(names, []byte(orSuffix))\n\t\t\t\t}\n\t\t\t\tnamePrefix = tf.regexpPrefix\n\t\t\t}\n\t\t} else if !tf.isNegative && !tf.isEmptyMatch {\n\t\t\thasPositiveFilter = true\n\t\t}\n\t}\n\tif len(names) == 0 {\n\t\tatomic.AddUint64(&compositeFilterMissingConversions, 1)\n\t\treturn []*TagFilters{tfs}\n\t}\n\n\t// Create composite filters for the found names.\n\tvar compositeKey, nameWithPrefix []byte\n\tfor _, name := range names {\n\t\tcompositeFilters := 0\n\t\ttfsNew := make([]tagFilter, 0, len(tfs.tfs))\n\t\tfor _, tf := range tfs.tfs {\n\t\t\tif len(tf.key) == 0 {\n\t\t\t\tif !hasPositiveFilter || tf.isNegative {\n\t\t\t\t\t// Negative filters on metric name cannot be used for building composite filter, so leave them as is.\n\t\t\t\t\ttfsNew = append(tfsNew, tf)\n\t\t\t\t\tcontinue\n\t\t\t\t}\n\t\t\t\tif tf.isRegexp {\n\t\t\t\t\tmatchName := false\n\t\t\t\t\tfor _, orSuffix := range tf.orSuffixes {\n\t\t\t\t\t\tif orSuffix == string(name) {\n\t\t\t\t\t\t\tmatchName = true\n\t\t\t\t\t\t\tbreak\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t\tif !matchName {\n\t\t\t\t\t\t// Leave as is the regexp filter on metric name if it doesn't match the current name.\n\t\t\t\t\t\ttfsNew = append(tfsNew, tf)\n\t\t\t\t\t\tcontinue\n\t\t\t\t\t}\n\t\t\t\t\t// Skip the tf, since its part (name) is used as a prefix in composite filter.\n\t\t\t\t\tcontinue\n\t\t\t\t}\n\t\t\t\tif string(tf.value) != string(name) {\n\t\t\t\t\t// Leave as is the filter on another metric name.\n\t\t\t\t\ttfsNew = append(tfsNew, tf)\n\t\t\t\t\tcontinue\n\t\t\t\t}\n\t\t\t\t// Skip the tf, since it is used as a prefix in composite filter.\n\t\t\t\tcontinue\n\t\t\t}\n\t\t\tif string(tf.key) == \"__graphite__\" || bytes.Equal(tf.key, graphiteReverseTagKey) {\n\t\t\t\t// Leave as is __graphite__ filters, since they cannot be used for building composite filter.\n\t\t\t\ttfsNew = append(tfsNew, tf)\n\t\t\t\tcontinue\n\t\t\t}\n\t\t\t// Create composite filter on (name, tf)\n\t\t\tnameWithPrefix = append(nameWithPrefix[:0], namePrefix...)\n\t\t\tnameWithPrefix = append(nameWithPrefix, name...)\n\t\t\tcompositeKey = marshalCompositeTagKey(compositeKey[:0], nameWithPrefix, tf.key)\n\t\t\tvar tfNew tagFilter\n\t\t\tif err := tfNew.Init(tfs.commonPrefix, compositeKey, tf.value, tf.isNegative, tf.isRegexp); err != nil {\n\t\t\t\tlogger.Panicf(\"BUG: unexpected error when creating composite tag filter for name=%q and key=%q: %s\", name, tf.key, err)\n\t\t\t}\n\t\t\ttfsNew = append(tfsNew, tfNew)\n\t\t\tcompositeFilters++\n\t\t}\n\t\tif compositeFilters == 0 {\n\t\t\t// Cannot use tfsNew, since it doesn't contain composite filters, e.g. it may match broader set of series.\n\t\t\t// Fall back to the original tfs.\n\t\t\tatomic.AddUint64(&compositeFilterMissingConversions, 1)\n\t\t\treturn []*TagFilters{tfs}\n\t\t}\n\t\ttfsCompiled := NewTagFilters(tfs.accountID, tfs.projectID)\n\t\ttfsCompiled.tfs = tfsNew\n\t\ttfssCompiled = append(tfssCompiled, tfsCompiled)\n\t}\n\tatomic.AddUint64(&compositeFilterSuccessConversions, 1)\n\treturn tfssCompiled\n}\n\nvar (\n\tcompositeFilterSuccessConversions uint64\n\tcompositeFilterMissingConversions uint64\n)\n\n// TagFilters represents filters used for filtering tags.\ntype TagFilters struct {\n\taccountID uint32\n\tprojectID uint32\n\n\ttfs []tagFilter\n\n\t// Common prefix for all the tag filters.\n\t// Contains encoded nsPrefixTagToMetricIDs + accountID + projectID.\n\tcommonPrefix []byte\n}\n\n// NewTagFilters returns new TagFilters for the given accountID and projectID.\nfunc NewTagFilters(accountID, projectID uint32) *TagFilters {\n\treturn &TagFilters{\n\t\taccountID:    accountID,\n\t\tprojectID:    projectID,\n\t\tcommonPrefix: marshalCommonPrefix(nil, nsPrefixTagToMetricIDs, accountID, projectID),\n\t}\n}\n\n// AddGraphiteQuery adds the given Graphite query that matches the given paths to tfs.\nfunc (tfs *TagFilters) AddGraphiteQuery(query []byte, paths []string, isNegative bool) {\n\ttf := tfs.addTagFilter()\n\ttf.InitFromGraphiteQuery(tfs.commonPrefix, query, paths, isNegative)\n}\n\n// Add adds the given tag filter to tfs.\n//\n// MetricGroup must be encoded with nil key.\nfunc (tfs *TagFilters) Add(key, value []byte, isNegative, isRegexp bool) error {\n\t// Verify whether tag filter is empty.\n\tif len(value) == 0 {\n\t\t// Substitute an empty tag value with the negative match\n\t\t// of `.+` regexp in order to filter out all the values with\n\t\t// the given tag.\n\t\tisNegative = !isNegative\n\t\tisRegexp = true\n\t\tvalue = []byte(\".+\")\n\t}\n\tif isRegexp && string(value) == \".*\" {\n\t\tif !isNegative {\n\t\t\t// Skip tag filter matching anything, since it equal to no filter.\n\t\t\treturn nil\n\t\t}\n\n\t\t// Substitute negative tag filter matching anything with negative tag filter matching non-empty value\n\t\t// in order to filter out all the time series with the given key.\n\t\tvalue = []byte(\".+\")\n\t}\n\n\ttf := tfs.addTagFilter()\n\tif err := tf.Init(tfs.commonPrefix, key, value, isNegative, isRegexp); err != nil {\n\t\treturn fmt.Errorf(\"cannot initialize tagFilter: %w\", err)\n\t}\n\tif tf.isNegative && tf.isEmptyMatch {\n\t\t// We have {key!~\"|foo\"} tag filter, which matches non=empty key values.\n\t\t// So add {key=~\".+\"} tag filter in order to enforce this.\n\t\t// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/546 for details.\n\t\ttfNew := tfs.addTagFilter()\n\t\tif err := tfNew.Init(tfs.commonPrefix, key, []byte(\".+\"), false, true); err != nil {\n\t\t\treturn fmt.Errorf(`cannot initialize {%s=\".+\"} tag filter: %w`, key, err)\n\t\t}\n\t}\n\tif len(tf.graphiteReverseSuffix) > 0 {\n\t\tre := regexp.QuoteMeta(string(tf.graphiteReverseSuffix)) + \".*\"\n\t\ttfNew := tfs.addTagFilter()\n\t\tif err := tfNew.Init(tfs.commonPrefix, graphiteReverseTagKey, []byte(re), false, true); err != nil {\n\t\t\treturn fmt.Errorf(\"cannot initialize reverse tag filter for Graphite wildcard: %w\", err)\n\t\t}\n\t}\n\treturn nil\n}\n\nfunc (tfs *TagFilters) addTagFilter() *tagFilter {\n\tif cap(tfs.tfs) > len(tfs.tfs) {\n\t\ttfs.tfs = tfs.tfs[:len(tfs.tfs)+1]\n\t} else {\n\t\ttfs.tfs = append(tfs.tfs, tagFilter{})\n\t}\n\treturn &tfs.tfs[len(tfs.tfs)-1]\n}\n\n// String returns human-readable value for tfs.\nfunc (tfs *TagFilters) String() string {\n\tvar bb bytes.Buffer\n\tfmt.Fprintf(&bb, \"AccountID=%d, ProjectID=%d \", tfs.accountID, tfs.projectID)\n\tif len(tfs.tfs) == 0 {\n\t\tfmt.Fprintf(&bb, \"{}\")\n\t\treturn bb.String()\n\t}\n\tfmt.Fprintf(&bb, \"{%s\", tfs.tfs[0].String())\n\tfor i := range tfs.tfs[1:] {\n\t\tfmt.Fprintf(&bb, \", %s\", tfs.tfs[i+1].String())\n\t}\n\tfmt.Fprintf(&bb, \"}\")\n\treturn bb.String()\n}\n\n// Reset resets the tf for the given accountID and projectID\nfunc (tfs *TagFilters) Reset(accountID, projectID uint32) {\n\ttfs.accountID = accountID\n\ttfs.projectID = projectID\n\ttfs.tfs = tfs.tfs[:0]\n\ttfs.commonPrefix = marshalCommonPrefix(tfs.commonPrefix[:0], nsPrefixTagToMetricIDs, accountID, projectID)\n}\n\n// tagFilter represents a filter used for filtering tags.\ntype tagFilter struct {\n\tkey        []byte\n\tvalue      []byte\n\tisNegative bool\n\tisRegexp   bool\n\n\t// matchCost is a cost for matching a filter against a single string.\n\tmatchCost uint64\n\n\t// contains the prefix for regexp filter if isRegexp==true.\n\tregexpPrefix string\n\n\t// Prefix always contains {nsPrefixTagToMetricIDs, AccountID, ProjectID, key}.\n\t// Additionally it contains:\n\t//  - value if !isRegexp.\n\t//  - regexpPrefix if isRegexp.\n\tprefix []byte\n\n\t// `or` values obtained from regexp suffix if it equals to \"foo|bar|...\"\n\t//\n\t// the regexp prefix is stored in regexpPrefix.\n\t//\n\t// This array is also populated with matching Graphite metrics if key=\"__graphite__\"\n\torSuffixes []string\n\n\t// Matches regexp suffix.\n\treSuffixMatch func(b []byte) bool\n\n\t// Set to true for filters matching empty value.\n\tisEmptyMatch bool\n\n\t// Contains reverse suffix for Graphite wildcard.\n\t// I.e. for `{__name__=~\"foo\\\\.[^.]*\\\\.bar\\\\.baz\"}` the value will be `zab.rab.`\n\tgraphiteReverseSuffix []byte\n}\n\nfunc (tf *tagFilter) isComposite() bool {\n\tk := tf.key\n\treturn len(k) > 0 && k[0] == compositeTagKeyPrefix\n}\n\nfunc (tf *tagFilter) Less(other *tagFilter) bool {\n\t// Move composite filters to the top, since they usually match lower number of time series.\n\t// Move regexp filters to the bottom, since they require scanning all the entries for the given label.\n\tisCompositeA := tf.isComposite()\n\tisCompositeB := tf.isComposite()\n\tif isCompositeA != isCompositeB {\n\t\treturn isCompositeA\n\t}\n\tif tf.matchCost != other.matchCost {\n\t\treturn tf.matchCost < other.matchCost\n\t}\n\tif tf.isRegexp != other.isRegexp {\n\t\treturn !tf.isRegexp\n\t}\n\tif len(tf.orSuffixes) != len(other.orSuffixes) {\n\t\treturn len(tf.orSuffixes) < len(other.orSuffixes)\n\t}\n\tif tf.isNegative != other.isNegative {\n\t\treturn !tf.isNegative\n\t}\n\treturn bytes.Compare(tf.prefix, other.prefix) < 0\n}\n\n// String returns human-readable tf value.\nfunc (tf *tagFilter) String() string {\n\top := \"=\"\n\tif tf.isNegative {\n\t\top = \"!=\"\n\t\tif tf.isRegexp {\n\t\t\top = \"!~\"\n\t\t}\n\t} else if tf.isRegexp {\n\t\top = \"=~\"\n\t}\n\tkey := tf.key\n\tif len(key) == 0 {\n\t\tkey = []byte(\"__name__\")\n\t}\n\treturn fmt.Sprintf(\"%s%s%q\", key, op, tf.value)\n}\n\nfunc (tf *tagFilter) Marshal(dst []byte, accountID, projectID uint32) []byte {\n\tdst = encoding.MarshalUint32(dst, accountID)\n\tdst = encoding.MarshalUint32(dst, projectID)\n\treturn tf.MarshalNoAccountIDProjectID(dst)\n}\n\n// MarshalNoAccountIDProjectID appends marshaled tf to dst\n// and returns the result.\nfunc (tf *tagFilter) MarshalNoAccountIDProjectID(dst []byte) []byte {\n\tdst = marshalTagValue(dst, tf.key)\n\tdst = marshalTagValue(dst, tf.value)\n\n\tisNegative := byte(0)\n\tif tf.isNegative {\n\t\tisNegative = 1\n\t}\n\n\tisRegexp := byte(0)\n\tif tf.isRegexp {\n\t\tisRegexp = 1\n\t}\n\n\tdst = append(dst, isNegative, isRegexp)\n\treturn dst\n}\n\n// InitFromGraphiteQuery initializes tf from the given graphite query expanded to the given paths.\nfunc (tf *tagFilter) InitFromGraphiteQuery(commonPrefix, query []byte, paths []string, isNegative bool) {\n\tif len(paths) == 0 {\n\t\t// explicitly add empty path in order match zero metric names.\n\t\tpaths = []string{\"\"}\n\t}\n\tprefix, orSuffixes := getCommonPrefix(paths)\n\tif len(orSuffixes) == 0 {\n\t\torSuffixes = append(orSuffixes[:0], \"\")\n\t}\n\t// Sort orSuffixes for faster seek later.\n\tsort.Strings(orSuffixes)\n\n\ttf.key = append(tf.key[:0], \"__graphite__\"...)\n\ttf.value = append(tf.value[:0], query...)\n\ttf.isNegative = isNegative\n\ttf.isRegexp = true // this is needed for tagFilter.matchSuffix\n\ttf.regexpPrefix = prefix\n\ttf.prefix = append(tf.prefix[:0], commonPrefix...)\n\ttf.prefix = marshalTagValue(tf.prefix, nil)\n\ttf.prefix = marshalTagValueNoTrailingTagSeparator(tf.prefix, []byte(prefix))\n\ttf.orSuffixes = append(tf.orSuffixes[:0], orSuffixes...)\n\ttf.reSuffixMatch, tf.matchCost = newMatchFuncForOrSuffixes(orSuffixes)\n}\n\nfunc getCommonPrefix(ss []string) (string, []string) {\n\tif len(ss) == 0 {\n\t\treturn \"\", nil\n\t}\n\tprefix := ss[0]\n\tfor _, s := range ss[1:] {\n\t\ti := 0\n\t\tfor i < len(s) && i < len(prefix) && s[i] == prefix[i] {\n\t\t\ti++\n\t\t}\n\t\tprefix = prefix[:i]\n\t\tif len(prefix) == 0 {\n\t\t\treturn \"\", ss\n\t\t}\n\t}\n\tresult := make([]string, len(ss))\n\tfor i, s := range ss {\n\t\tresult[i] = s[len(prefix):]\n\t}\n\treturn prefix, result\n}\n\n// Init initializes the tag filter for the given commonPrefix, key and value.\n//\n// If isNegaitve is true, then the tag filter matches all the values\n// except the given one.\n//\n// If isRegexp is true, then the value is interpreted as anchored regexp,\n// i.e. '^(tag.Value)$'.\n//\n// MetricGroup must be encoded in the value with nil key.\nfunc (tf *tagFilter) Init(commonPrefix, key, value []byte, isNegative, isRegexp bool) error {\n\ttf.key = append(tf.key[:0], key...)\n\ttf.value = append(tf.value[:0], value...)\n\ttf.isNegative = isNegative\n\ttf.isRegexp = isRegexp\n\ttf.matchCost = 0\n\n\ttf.regexpPrefix = \"\"\n\ttf.prefix = tf.prefix[:0]\n\n\ttf.orSuffixes = tf.orSuffixes[:0]\n\ttf.reSuffixMatch = nil\n\ttf.isEmptyMatch = false\n\ttf.graphiteReverseSuffix = tf.graphiteReverseSuffix[:0]\n\n\ttf.prefix = append(tf.prefix, commonPrefix...)\n\ttf.prefix = marshalTagValue(tf.prefix, key)\n\n\tvar expr []byte\n\tprefix := tf.value\n\tif tf.isRegexp {\n\t\tprefix, expr = getRegexpPrefix(tf.value)\n\t\tif len(expr) == 0 {\n\t\t\ttf.value = append(tf.value[:0], prefix...)\n\t\t\ttf.isRegexp = false\n\t\t} else {\n\t\t\ttf.regexpPrefix = string(prefix)\n\t\t}\n\t}\n\ttf.prefix = marshalTagValueNoTrailingTagSeparator(tf.prefix, prefix)\n\tif !tf.isRegexp {\n\t\t// tf contains plain value without regexp.\n\t\t// Add empty orSuffix in order to trigger fast path for orSuffixes\n\t\t// during the search for matching metricIDs.\n\t\ttf.orSuffixes = append(tf.orSuffixes[:0], \"\")\n\t\ttf.isEmptyMatch = len(prefix) == 0\n\t\ttf.matchCost = fullMatchCost\n\t\treturn nil\n\t}\n\trcv, err := getRegexpFromCache(expr)\n\tif err != nil {\n\t\treturn err\n\t}\n\ttf.orSuffixes = append(tf.orSuffixes[:0], rcv.orValues...)\n\ttf.reSuffixMatch = rcv.reMatch\n\ttf.matchCost = rcv.reCost\n\ttf.isEmptyMatch = len(prefix) == 0 && tf.reSuffixMatch(nil)\n\tif !tf.isNegative && len(key) == 0 && strings.IndexByte(rcv.literalSuffix, '.') >= 0 {\n\t\t// Reverse suffix is needed only for non-negative regexp filters on __name__ that contains dots.\n\t\ttf.graphiteReverseSuffix = reverseBytes(tf.graphiteReverseSuffix[:0], []byte(rcv.literalSuffix))\n\t}\n\treturn nil\n}\n\nfunc (tf *tagFilter) match(b []byte) (bool, error) {\n\tprefix := tf.prefix\n\tif !bytes.HasPrefix(b, prefix) {\n\t\treturn tf.isNegative, nil\n\t}\n\tok, err := tf.matchSuffix(b[len(prefix):])\n\tif err != nil {\n\t\treturn false, err\n\t}\n\tif !ok {\n\t\treturn tf.isNegative, nil\n\t}\n\treturn !tf.isNegative, nil\n}\n\nfunc (tf *tagFilter) matchSuffix(b []byte) (bool, error) {\n\t// Remove the trailing tagSeparatorChar.\n\tif len(b) == 0 || b[len(b)-1] != tagSeparatorChar {\n\t\treturn false, fmt.Errorf(\"unexpected end of b; want %d; b=%q\", tagSeparatorChar, b)\n\t}\n\tb = b[:len(b)-1]\n\tif !tf.isRegexp {\n\t\treturn len(b) == 0, nil\n\t}\n\tok := tf.reSuffixMatch(b)\n\treturn ok, nil\n}\n\n// RegexpCacheSize returns the number of cached regexps for tag filters.\nfunc RegexpCacheSize() int {\n\tregexpCacheLock.RLock()\n\tn := len(regexpCacheMap)\n\tregexpCacheLock.RUnlock()\n\treturn n\n}\n\n// RegexpCacheRequests returns the number of requests to regexp cache.\nfunc RegexpCacheRequests() uint64 {\n\treturn atomic.LoadUint64(&regexpCacheRequests)\n}\n\n// RegexpCacheMisses returns the number of cache misses for regexp cache.\nfunc RegexpCacheMisses() uint64 {\n\treturn atomic.LoadUint64(&regexpCacheMisses)\n}\n\nfunc getRegexpFromCache(expr []byte) (regexpCacheValue, error) {\n\tatomic.AddUint64(&regexpCacheRequests, 1)\n\n\tregexpCacheLock.RLock()\n\trcv, ok := regexpCacheMap[string(expr)]\n\tregexpCacheLock.RUnlock()\n\tif ok {\n\t\t// Fast path - the regexp found in the cache.\n\t\treturn rcv, nil\n\t}\n\n\t// Slow path - build the regexp.\n\tatomic.AddUint64(&regexpCacheMisses, 1)\n\texprOrig := string(expr)\n\n\texpr = []byte(tagCharsRegexpEscaper.Replace(exprOrig))\n\texprStr := fmt.Sprintf(\"^(%s)$\", expr)\n\tre, err := regexp.Compile(exprStr)\n\tif err != nil {\n\t\treturn rcv, fmt.Errorf(\"invalid regexp %q: %w\", exprStr, err)\n\t}\n\n\tsExpr := string(expr)\n\torValues := getOrValues(sExpr)\n\tvar reMatch func(b []byte) bool\n\tvar reCost uint64\n\tvar literalSuffix string\n\tif len(orValues) > 0 {\n\t\treMatch, reCost = newMatchFuncForOrSuffixes(orValues)\n\t} else {\n\t\treMatch, literalSuffix, reCost = getOptimizedReMatchFunc(re.Match, sExpr)\n\t}\n\n\t// Put the reMatch in the cache.\n\trcv.orValues = orValues\n\trcv.reMatch = reMatch\n\trcv.reCost = reCost\n\trcv.literalSuffix = literalSuffix\n\n\tregexpCacheLock.Lock()\n\tif overflow := len(regexpCacheMap) - getMaxRegexpCacheSize(); overflow > 0 {\n\t\toverflow = int(float64(len(regexpCacheMap)) * 0.1)\n\t\tfor k := range regexpCacheMap {\n\t\t\tdelete(regexpCacheMap, k)\n\t\t\toverflow--\n\t\t\tif overflow <= 0 {\n\t\t\t\tbreak\n\t\t\t}\n\t\t}\n\t}\n\tregexpCacheMap[exprOrig] = rcv\n\tregexpCacheLock.Unlock()\n\n\treturn rcv, nil\n}\n\nfunc newMatchFuncForOrSuffixes(orValues []string) (reMatch func(b []byte) bool, reCost uint64) {\n\tif len(orValues) == 1 {\n\t\tv := orValues[0]\n\t\treMatch = func(b []byte) bool {\n\t\t\treturn string(b) == v\n\t\t}\n\t} else {\n\t\treMatch = func(b []byte) bool {\n\t\t\tfor _, v := range orValues {\n\t\t\t\tif string(b) == v {\n\t\t\t\t\treturn true\n\t\t\t\t}\n\t\t\t}\n\t\t\treturn false\n\t\t}\n\t}\n\treCost = uint64(len(orValues)) * literalMatchCost\n\treturn reMatch, reCost\n}\n\n// getOptimizedReMatchFunc tries returning optimized function for matching the given expr.\n//\n//\t'.*'\n//\t'.+'\n//\t'literal.*'\n//\t'literal.+'\n//\t'.*literal'\n//\t'.+literal\n//\t'.*literal.*'\n//\t'.*literal.+'\n//\t'.+literal.*'\n//\t'.+literal.+'\n//\n// It returns reMatch if it cannot find optimized function.\n//\n// It also returns literal suffix from the expr.\nfunc getOptimizedReMatchFunc(reMatch func(b []byte) bool, expr string) (func(b []byte) bool, string, uint64) {\n\tsre, err := syntax.Parse(expr, syntax.Perl)\n\tif err != nil {\n\t\tlogger.Panicf(\"BUG: unexpected error when parsing verified expr=%q: %s\", expr, err)\n\t}\n\tif matchFunc, literalSuffix, reCost := getOptimizedReMatchFuncExt(reMatch, sre); matchFunc != nil {\n\t\t// Found optimized function for matching the expr.\n\t\tsuffixUnescaped := tagCharsReverseRegexpEscaper.Replace(literalSuffix)\n\t\treturn matchFunc, suffixUnescaped, reCost\n\t}\n\t// Fall back to un-optimized reMatch.\n\treturn reMatch, \"\", reMatchCost\n}\n\n// These cost values are used for sorting tag filters in ascending order or the required CPU time for execution.\n//\n// These values are obtained from BenchmarkOptimizedReMatchCost benchmark.\nconst (\n\tfullMatchCost    = 1\n\tprefixMatchCost  = 2\n\tliteralMatchCost = 3\n\tsuffixMatchCost  = 4\n\tmiddleMatchCost  = 6\n\treMatchCost      = 100\n)\n\nfunc getOptimizedReMatchFuncExt(reMatch func(b []byte) bool, sre *syntax.Regexp) (func(b []byte) bool, string, uint64) {\n\tif isDotStar(sre) {\n\t\t// '.*'\n\t\treturn func(b []byte) bool {\n\t\t\treturn true\n\t\t}, \"\", fullMatchCost\n\t}\n\tif isDotPlus(sre) {\n\t\t// '.+'\n\t\treturn func(b []byte) bool {\n\t\t\treturn len(b) > 0\n\t\t}, \"\", fullMatchCost\n\t}\n\tswitch sre.Op {\n\tcase syntax.OpCapture:\n\t\t// Remove parenthesis from expr, i.e. '(expr) -> expr'\n\t\treturn getOptimizedReMatchFuncExt(reMatch, sre.Sub[0])\n\tcase syntax.OpLiteral:\n\t\tif !isLiteral(sre) {\n\t\t\treturn nil, \"\", 0\n\t\t}\n\t\ts := string(sre.Rune)\n\t\t// Literal match\n\t\treturn func(b []byte) bool {\n\t\t\treturn string(b) == s\n\t\t}, s, literalMatchCost\n\tcase syntax.OpConcat:\n\t\tif len(sre.Sub) == 2 {\n\t\t\tif isLiteral(sre.Sub[0]) {\n\t\t\t\tprefix := []byte(string(sre.Sub[0].Rune))\n\t\t\t\tif isDotStar(sre.Sub[1]) {\n\t\t\t\t\t// 'prefix.*'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn bytes.HasPrefix(b, prefix)\n\t\t\t\t\t}, \"\", prefixMatchCost\n\t\t\t\t}\n\t\t\t\tif isDotPlus(sre.Sub[1]) {\n\t\t\t\t\t// 'prefix.+'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn len(b) > len(prefix) && bytes.HasPrefix(b, prefix)\n\t\t\t\t\t}, \"\", prefixMatchCost\n\t\t\t\t}\n\t\t\t}\n\t\t\tif isLiteral(sre.Sub[1]) {\n\t\t\t\tsuffix := []byte(string(sre.Sub[1].Rune))\n\t\t\t\tif isDotStar(sre.Sub[0]) {\n\t\t\t\t\t// '.*suffix'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn bytes.HasSuffix(b, suffix)\n\t\t\t\t\t}, string(suffix), suffixMatchCost\n\t\t\t\t}\n\t\t\t\tif isDotPlus(sre.Sub[0]) {\n\t\t\t\t\t// '.+suffix'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn len(b) > len(suffix) && bytes.HasSuffix(b[1:], suffix)\n\t\t\t\t\t}, string(suffix), suffixMatchCost\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t\tif len(sre.Sub) == 3 && isLiteral(sre.Sub[1]) {\n\t\t\tmiddle := []byte(string(sre.Sub[1].Rune))\n\t\t\tif isDotStar(sre.Sub[0]) {\n\t\t\t\tif isDotStar(sre.Sub[2]) {\n\t\t\t\t\t// '.*middle.*'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn bytes.Contains(b, middle)\n\t\t\t\t\t}, \"\", middleMatchCost\n\t\t\t\t}\n\t\t\t\tif isDotPlus(sre.Sub[2]) {\n\t\t\t\t\t// '.*middle.+'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn len(b) > len(middle) && bytes.Contains(b[:len(b)-1], middle)\n\t\t\t\t\t}, \"\", middleMatchCost\n\t\t\t\t}\n\t\t\t}\n\t\t\tif isDotPlus(sre.Sub[0]) {\n\t\t\t\tif isDotStar(sre.Sub[2]) {\n\t\t\t\t\t// '.+middle.*'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn len(b) > len(middle) && bytes.Contains(b[1:], middle)\n\t\t\t\t\t}, \"\", middleMatchCost\n\t\t\t\t}\n\t\t\t\tif isDotPlus(sre.Sub[2]) {\n\t\t\t\t\t// '.+middle.+'\n\t\t\t\t\treturn func(b []byte) bool {\n\t\t\t\t\t\treturn len(b) > len(middle)+1 && bytes.Contains(b[1:len(b)-1], middle)\n\t\t\t\t\t}, \"\", middleMatchCost\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t\t// Verify that the string matches all the literals found in the regexp\n\t\t// before applying the regexp.\n\t\t// This should optimize the case when the regexp doesn't match the string.\n\t\tvar literals [][]byte\n\t\tfor _, sub := range sre.Sub {\n\t\t\tif isLiteral(sub) {\n\t\t\t\tliterals = append(literals, []byte(string(sub.Rune)))\n\t\t\t}\n\t\t}\n\t\tvar suffix []byte\n\t\tif isLiteral(sre.Sub[len(sre.Sub)-1]) {\n\t\t\tsuffix = literals[len(literals)-1]\n\t\t\tliterals = literals[:len(literals)-1]\n\t\t}\n\t\treturn func(b []byte) bool {\n\t\t\tif len(suffix) > 0 && !bytes.HasSuffix(b, suffix) {\n\t\t\t\t// Fast path - b has no the given suffix\n\t\t\t\treturn false\n\t\t\t}\n\t\t\tbOrig := b\n\t\t\tfor _, literal := range literals {\n\t\t\t\tn := bytes.Index(b, literal)\n\t\t\t\tif n < 0 {\n\t\t\t\t\t// Fast path - b doesn't match the regexp.\n\t\t\t\t\treturn false\n\t\t\t\t}\n\t\t\t\tb = b[n+len(literal):]\n\t\t\t}\n\t\t\t// Fall back to slow path.\n\t\t\treturn reMatch(bOrig)\n\t\t}, string(suffix), reMatchCost\n\tdefault:\n\t\treturn nil, \"\", 0\n\t}\n}\n\nfunc isDotStar(sre *syntax.Regexp) bool {\n\tswitch sre.Op {\n\tcase syntax.OpCapture:\n\t\treturn isDotStar(sre.Sub[0])\n\tcase syntax.OpAlternate:\n\t\tfor _, reSub := range sre.Sub {\n\t\t\tif isDotStar(reSub) {\n\t\t\t\treturn true\n\t\t\t}\n\t\t}\n\t\treturn false\n\tcase syntax.OpStar:\n\t\tswitch sre.Sub[0].Op {\n\t\tcase syntax.OpAnyCharNotNL, syntax.OpAnyChar:\n\t\t\treturn true\n\t\tdefault:\n\t\t\treturn false\n\t\t}\n\tdefault:\n\t\treturn false\n\t}\n}\n\nfunc isDotPlus(sre *syntax.Regexp) bool {\n\tswitch sre.Op {\n\tcase syntax.OpCapture:\n\t\treturn isDotPlus(sre.Sub[0])\n\tcase syntax.OpAlternate:\n\t\tfor _, reSub := range sre.Sub {\n\t\t\tif isDotPlus(reSub) {\n\t\t\t\treturn true\n\t\t\t}\n\t\t}\n\t\treturn false\n\tcase syntax.OpPlus:\n\t\tswitch sre.Sub[0].Op {\n\t\tcase syntax.OpAnyCharNotNL, syntax.OpAnyChar:\n\t\t\treturn true\n\t\tdefault:\n\t\t\treturn false\n\t\t}\n\tdefault:\n\t\treturn false\n\t}\n}\n\nfunc isLiteral(sre *syntax.Regexp) bool {\n\tif sre.Op == syntax.OpCapture {\n\t\treturn isLiteral(sre.Sub[0])\n\t}\n\treturn sre.Op == syntax.OpLiteral && sre.Flags&syntax.FoldCase == 0\n}\n\nfunc getOrValues(expr string) []string {\n\tsre, err := syntax.Parse(expr, syntax.Perl)\n\tif err != nil {\n\t\tlogger.Panicf(\"BUG: unexpected error when parsing verified expr=%q: %s\", expr, err)\n\t}\n\torValues := getOrValuesExt(sre)\n\n\t// Sort orValues for faster index seek later\n\tsort.Strings(orValues)\n\n\treturn orValues\n}\n\nfunc getOrValuesExt(sre *syntax.Regexp) []string {\n\tswitch sre.Op {\n\tcase syntax.OpCapture:\n\t\treturn getOrValuesExt(sre.Sub[0])\n\tcase syntax.OpLiteral:\n\t\tif !isLiteral(sre) {\n\t\t\treturn nil\n\t\t}\n\t\treturn []string{string(sre.Rune)}\n\tcase syntax.OpEmptyMatch:\n\t\treturn []string{\"\"}\n\tcase syntax.OpAlternate:\n\t\ta := make([]string, 0, len(sre.Sub))\n\t\tfor _, reSub := range sre.Sub {\n\t\t\tca := getOrValuesExt(reSub)\n\t\t\tif len(ca) == 0 {\n\t\t\t\treturn nil\n\t\t\t}\n\t\t\ta = append(a, ca...)\n\t\t\tif len(a) > maxOrValues {\n\t\t\t\t// It is cheaper to use regexp here.\n\t\t\t\treturn nil\n\t\t\t}\n\t\t}\n\t\treturn a\n\tcase syntax.OpCharClass:\n\t\ta := make([]string, 0, len(sre.Rune)/2)\n\t\tfor i := 0; i < len(sre.Rune); i += 2 {\n\t\t\tstart := sre.Rune[i]\n\t\t\tend := sre.Rune[i+1]\n\t\t\tfor start <= end {\n\t\t\t\ta = append(a, string(start))\n\t\t\t\tstart++\n\t\t\t\tif len(a) > maxOrValues {\n\t\t\t\t\t// It is cheaper to use regexp here.\n\t\t\t\t\treturn nil\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t\treturn a\n\tcase syntax.OpConcat:\n\t\tif len(sre.Sub) < 1 {\n\t\t\treturn []string{\"\"}\n\t\t}\n\t\tprefixes := getOrValuesExt(sre.Sub[0])\n\t\tif len(prefixes) == 0 {\n\t\t\treturn nil\n\t\t}\n\t\tsre.Sub = sre.Sub[1:]\n\t\tsuffixes := getOrValuesExt(sre)\n\t\tif len(suffixes) == 0 {\n\t\t\treturn nil\n\t\t}\n\t\tif len(prefixes)*len(suffixes) > maxOrValues {\n\t\t\t// It is cheaper to use regexp here.\n\t\t\treturn nil\n\t\t}\n\t\ta := make([]string, 0, len(prefixes)*len(suffixes))\n\t\tfor _, prefix := range prefixes {\n\t\t\tfor _, suffix := range suffixes {\n\t\t\t\ts := prefix + suffix\n\t\t\t\ta = append(a, s)\n\t\t\t}\n\t\t}\n\t\treturn a\n\tdefault:\n\t\treturn nil\n\t}\n}\n\nconst maxOrValues = 20\n\nvar tagCharsRegexpEscaper = strings.NewReplacer(\n\t\"\\\\x00\", \"\\\\x000\", // escapeChar\n\t\"\\x00\", \"\\\\x000\", // escapeChar\n\t\"\\\\x01\", \"\\\\x001\", // tagSeparatorChar\n\t\"\\x01\", \"\\\\x001\", // tagSeparatorChar\n\t\"\\\\x02\", \"\\\\x002\", // kvSeparatorChar\n\t\"\\x02\", \"\\\\x002\", // kvSeparatorChar\n)\n\nvar tagCharsReverseRegexpEscaper = strings.NewReplacer(\n\t\"\\\\x000\", \"\\x00\", // escapeChar\n\t\"\\x000\", \"\\x00\", // escapeChar\n\t\"\\\\x001\", \"\\x01\", // tagSeparatorChar\n\t\"\\x001\", \"\\x01\", // tagSeparatorChar\n\t\"\\\\x002\", \"\\x02\", // kvSeparatorChar\n\t\"\\x002\", \"\\x02\", // kvSeparatorChar\n)\n\nfunc getMaxRegexpCacheSize() int {\n\tmaxRegexpCacheSizeOnce.Do(func() {\n\t\tn := memory.Allowed() / 1024 / 1024\n\t\tif n < 100 {\n\t\t\tn = 100\n\t\t}\n\t\tmaxRegexpCacheSize = n\n\t})\n\treturn maxRegexpCacheSize\n}\n\nvar (\n\tmaxRegexpCacheSize     int\n\tmaxRegexpCacheSizeOnce sync.Once\n)\n\nvar (\n\tregexpCacheMap  = make(map[string]regexpCacheValue)\n\tregexpCacheLock sync.RWMutex\n\n\tregexpCacheRequests uint64\n\tregexpCacheMisses   uint64\n)\n\ntype regexpCacheValue struct {\n\torValues      []string\n\treMatch       func(b []byte) bool\n\treCost        uint64\n\tliteralSuffix string\n}\n\nfunc getRegexpPrefix(b []byte) ([]byte, []byte) {\n\t// Fast path - search the prefix in the cache.\n\tprefixesCacheLock.RLock()\n\tps, ok := prefixesCacheMap[string(b)]\n\tprefixesCacheLock.RUnlock()\n\n\tif ok {\n\t\treturn ps.prefix, ps.suffix\n\t}\n\n\t// Slow path - extract the regexp prefix from b.\n\tprefix, suffix := extractRegexpPrefix(b)\n\n\t// Put the prefix and the suffix to the cache.\n\tprefixesCacheLock.Lock()\n\tif overflow := len(prefixesCacheMap) - getMaxPrefixesCacheSize(); overflow > 0 {\n\t\toverflow = int(float64(len(prefixesCacheMap)) * 0.1)\n\t\tfor k := range prefixesCacheMap {\n\t\t\tdelete(prefixesCacheMap, k)\n\t\t\toverflow--\n\t\t\tif overflow <= 0 {\n\t\t\t\tbreak\n\t\t\t}\n\t\t}\n\t}\n\tprefixesCacheMap[string(b)] = prefixSuffix{\n\t\tprefix: prefix,\n\t\tsuffix: suffix,\n\t}\n\tprefixesCacheLock.Unlock()\n\n\treturn prefix, suffix\n}\n\nfunc getMaxPrefixesCacheSize() int {\n\tmaxPrefixesCacheSizeOnce.Do(func() {\n\t\tn := memory.Allowed() / 1024 / 1024\n\t\tif n < 100 {\n\t\t\tn = 100\n\t\t}\n\t\tmaxPrefixesCacheSize = n\n\t})\n\treturn maxPrefixesCacheSize\n}\n\nvar (\n\tmaxPrefixesCacheSize     int\n\tmaxPrefixesCacheSizeOnce sync.Once\n)\n\nvar (\n\tprefixesCacheMap  = make(map[string]prefixSuffix)\n\tprefixesCacheLock sync.RWMutex\n)\n\ntype prefixSuffix struct {\n\tprefix []byte\n\tsuffix []byte\n}\n\nfunc extractRegexpPrefix(b []byte) ([]byte, []byte) {\n\tsre, err := syntax.Parse(string(b), syntax.Perl)\n\tif err != nil {\n\t\t// Cannot parse the regexp. Return it all as prefix.\n\t\treturn b, nil\n\t}\n\tsre = simplifyRegexp(sre)\n\tif sre == emptyRegexp {\n\t\treturn nil, nil\n\t}\n\tif isLiteral(sre) {\n\t\treturn []byte(string(sre.Rune)), nil\n\t}\n\tvar prefix []byte\n\tif sre.Op == syntax.OpConcat {\n\t\tsub0 := sre.Sub[0]\n\t\tif isLiteral(sub0) {\n\t\t\tprefix = []byte(string(sub0.Rune))\n\t\t\tsre.Sub = sre.Sub[1:]\n\t\t\tif len(sre.Sub) == 0 {\n\t\t\t\treturn nil, nil\n\t\t\t}\n\t\t}\n\t}\n\tif _, err := syntax.Compile(sre); err != nil {\n\t\t// Cannot compile the regexp. Return it all as prefix.\n\t\treturn b, nil\n\t}\n\treturn prefix, []byte(sre.String())\n}\n\nfunc simplifyRegexp(sre *syntax.Regexp) *syntax.Regexp {\n\ts := sre.String()\n\tfor {\n\t\tsre = simplifyRegexpExt(sre, false, false)\n\t\tsre = sre.Simplify()\n\t\tif sre.Op == syntax.OpBeginText || sre.Op == syntax.OpEndText {\n\t\t\tsre = emptyRegexp\n\t\t}\n\t\tsNew := sre.String()\n\t\tif sNew == s {\n\t\t\treturn sre\n\t\t}\n\t\tvar err error\n\t\tsre, err = syntax.Parse(sNew, syntax.Perl)\n\t\tif err != nil {\n\t\t\tlogger.Panicf(\"BUG: cannot parse simplified regexp %q: %s\", sNew, err)\n\t\t}\n\t\ts = sNew\n\t}\n}\n\nfunc simplifyRegexpExt(sre *syntax.Regexp, hasPrefix, hasSuffix bool) *syntax.Regexp {\n\tswitch sre.Op {\n\tcase syntax.OpCapture:\n\t\t// Substitute all the capture regexps with non-capture regexps.\n\t\tsre.Op = syntax.OpAlternate\n\t\tsre.Sub[0] = simplifyRegexpExt(sre.Sub[0], hasPrefix, hasSuffix)\n\t\tif sre.Sub[0] == emptyRegexp {\n\t\t\treturn emptyRegexp\n\t\t}\n\t\treturn sre\n\tcase syntax.OpStar, syntax.OpPlus, syntax.OpQuest, syntax.OpRepeat:\n\t\tsre.Sub[0] = simplifyRegexpExt(sre.Sub[0], hasPrefix, hasSuffix)\n\t\tif sre.Sub[0] == emptyRegexp {\n\t\t\treturn emptyRegexp\n\t\t}\n\t\treturn sre\n\tcase syntax.OpAlternate:\n\t\t// Do not remove empty captures from OpAlternate, since this may break regexp.\n\t\tfor i, sub := range sre.Sub {\n\t\t\tsre.Sub[i] = simplifyRegexpExt(sub, hasPrefix, hasSuffix)\n\t\t}\n\t\treturn sre\n\tcase syntax.OpConcat:\n\t\tsubs := sre.Sub[:0]\n\t\tfor i, sub := range sre.Sub {\n\t\t\tif sub = simplifyRegexpExt(sub, i > 0, i+1 < len(sre.Sub)); sub != emptyRegexp {\n\t\t\t\tsubs = append(subs, sub)\n\t\t\t}\n\t\t}\n\t\tsre.Sub = subs\n\t\t// Remove anchros from the beginning and the end of regexp, since they\n\t\t// will be added later.\n\t\tif !hasPrefix {\n\t\t\tfor len(sre.Sub) > 0 && sre.Sub[0].Op == syntax.OpBeginText {\n\t\t\t\tsre.Sub = sre.Sub[1:]\n\t\t\t}\n\t\t}\n\t\tif !hasSuffix {\n\t\t\tfor len(sre.Sub) > 0 && sre.Sub[len(sre.Sub)-1].Op == syntax.OpEndText {\n\t\t\t\tsre.Sub = sre.Sub[:len(sre.Sub)-1]\n\t\t\t}\n\t\t}\n\t\tif len(sre.Sub) == 0 {\n\t\t\treturn emptyRegexp\n\t\t}\n\t\treturn sre\n\tcase syntax.OpEmptyMatch:\n\t\treturn emptyRegexp\n\tdefault:\n\t\treturn sre\n\t}\n}\n\nvar emptyRegexp = &syntax.Regexp{\n\tOp: syntax.OpEmptyMatch,\n}\n\n```\n\n## Update code snippets\n\n```go\nfunc newMatchFuncForOrSuffixes(orValues []string) (reMatch func(b []byte) bool, reCost uint64) {\n\tif len(orValues) == 1 {\n\t\tv := orValues[0]\n\t\treMatch = func(b []byte) bool {\n\t\t\treturn string(b) == v\n\t\t}\n\t\treCost = literalMatchCost\n\t\treturn reMatch, reCost\n\t}\n\treMatch = func(b []byte) bool {\n\t\tfor _, v := range orValues {\n\t\t\tif string(b) == v {\n\t\t\t\treturn true\n\t\t\t}\n\t\t}\n\t\treturn false\n\t}\n\treCost = uint64(len(orValues)) * literalMatchCost\n\treturn reMatch, reCost\n}\n```\n\nPlease output the complete merged code, it's about 30324 bytes in total."
		}
	]
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
