// Years refer to the first arXiv release; venue includes the conference year.
// Metadata and reading guides checked against the linked primary sources.
window.STUDY_PAPERS = [
    {
        id: 'attention',
        title: 'Attention Is All You Need',
        shortTitle: 'Transformer · 注意力架构',
        year: 2017,
        category: 'architecture',
        authors: 'Vaswani 等',
        venue: 'NeurIPS 2017',
        url: 'https://arxiv.org/abs/1706.03762',
        pdf: 'https://arxiv.org/pdf/1706.03762',
        summary: '用自注意力和前馈网络构建编码器与解码器，替代序列建模中的循环和卷积。',
        focus: [
            '沿图 1 走一遍输入到输出：编码器、解码器分别接收什么？',
            '写出 Q、K、V 的形状，解释缩放点积和多头拆分各解决什么问题。',
            '对照解码器掩码与位置编码，区分训练时的并行计算和生成时的逐步输出。'
        ],
        takeaway: '注意力负责在位置之间交换信息；位置编码、前馈网络、残差连接共同构成完整的 Transformer。',
        limitation: '主要实验是机器翻译和句法分析，不能直接用这些结果判断现代聊天模型的能力。'
    },
    {
        id: 'bert',
        title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
        shortTitle: 'BERT · 双向预训练',
        year: 2018,
        category: 'architecture',
        authors: 'Devlin 等',
        venue: 'NAACL 2019',
        url: 'https://arxiv.org/abs/1810.04805',
        pdf: 'https://arxiv.org/pdf/1810.04805',
        summary: '通过掩码语言模型和下一句预测预训练双向编码器，再微调到具体的语言理解任务。',
        focus: [
            '解释掩码语言模型为何能同时利用左右文，它与逐词预测有什么不同？',
            '画出 token、句段和位置三种嵌入，并说明 [CLS] 与 [SEP] 的用途。',
            '对照预训练任务的消融实验，区分双向上下文和下一句预测的贡献。'
        ],
        takeaway: '先学习可迁移的双向文本表示，再用少量任务结构改动接入分类、问答等下游任务。',
        limitation: '原始 BERT 面向语言理解；阅读其双向编码方式时，应与自回归文本生成区分。'
    },
    {
        id: 'lora',
        title: 'LoRA: Low-Rank Adaptation of Large Language Models',
        shortTitle: 'LoRA · 低秩微调',
        year: 2021,
        category: 'tuning',
        authors: 'Hu 等',
        venue: 'ICLR 2022',
        url: 'https://arxiv.org/abs/2106.09685',
        pdf: 'https://arxiv.org/pdf/2106.09685',
        summary: '冻结预训练权重，用可训练的低秩矩阵表示权重增量，减少任务适配时的可训练参数。',
        focus: [
            '从一个线性层写出原权重与低秩增量，核算新增的可训练参数量。',
            '哪些权重被冻结、哪些矩阵会更新？推理前如何把增量合并回原权重？',
            '比较不同秩和不同目标矩阵的实验，检查参数量与任务效果的关系。'
        ],
        takeaway: '将任务适配集中到低秩权重增量中，多个任务可以共享同一份冻结的基础模型。',
        limitation: '低秩假设是否适合新任务，需要结合秩、目标层和任务数据重新验证。'
    },
    {
        id: 'qlora',
        title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
        shortTitle: 'QLoRA · 量化微调',
        year: 2023,
        category: 'tuning',
        authors: 'Dettmers 等',
        venue: 'NeurIPS 2023',
        url: 'https://arxiv.org/abs/2305.14314',
        pdf: 'https://arxiv.org/pdf/2305.14314',
        summary: '在冻结的 4 位量化基础模型上训练 LoRA，并通过量化和分页优化器控制显存占用。',
        focus: [
            '分别标出基础权重的存储精度、计算精度和适配器的更新路径。',
            'NF4、双重量化、分页优化器分别针对哪一部分显存开销？',
            '对照数据集与评估方式，检查聊天模型分数能支持多大范围的结论。'
        ],
        takeaway: '把基础权重量化与低秩适配结合；理解节省显存的来源，需要区分权重存储和训练计算。',
        limitation: '论文指出聊天评测存在局限；特定基准上的表现不应直接等同于全面的模型能力。'
    },
    {
        id: 'instructgpt',
        title: 'Training language models to follow instructions with human feedback',
        shortTitle: 'InstructGPT · 人类反馈',
        year: 2022,
        category: 'alignment',
        authors: 'Ouyang 等',
        venue: 'NeurIPS 2022',
        url: 'https://arxiv.org/abs/2203.02155',
        pdf: 'https://arxiv.org/pdf/2203.02155',
        summary: '先用人工示范微调，再用回答排序学习奖励，通过人类反馈强化学习改进指令遵循。',
        focus: [
            '把示范数据、偏好排序、奖励模型和策略优化连成一张训练流程图。',
            '分别说明监督微调与强化学习阶段的数据、训练目标和更新对象。',
            '阅读人工评估设置：谁在评分、使用什么提示分布、评价什么行为？'
        ],
        takeaway: '指令遵循依赖训练目标与反馈数据；人工示范和回答偏好在不同阶段承担不同作用。',
        limitation: '模型仍会犯错；论文中的人类偏好结果需要连同标注方式和提示分布一起理解。'
    },
    {
        id: 'dpo',
        title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
        shortTitle: 'DPO · 直接偏好优化',
        year: 2023,
        category: 'alignment',
        authors: 'Rafailov 等',
        venue: 'NeurIPS 2023',
        url: 'https://arxiv.org/abs/2305.18290',
        pdf: 'https://arxiv.org/pdf/2305.18290',
        summary: '把带约束的偏好学习目标转化为分类损失，直接用成对偏好数据训练语言模型。',
        focus: [
            '对照 RLHF 流程，找出 DPO 省去了哪些训练组件，又保留了哪些输入。',
            '沿奖励与策略的关系推导损失，解释参考模型和偏好回答各自的作用。',
            '检查实验中的偏好数据与任务范围，思考更换数据后应补充哪些评估。'
        ],
        takeaway: '通过奖励函数与最优策略之间的关系，可以直接优化偏好数据上的损失，无须单独训练奖励模型。',
        limitation: '偏好学习的目标不等于事实正确；迁移到新任务时仍需检查偏好数据质量与评估范围。'
    },
    {
        id: 'react',
        title: 'ReAct: Synergizing Reasoning and Acting in Language Models',
        shortTitle: 'ReAct · 推理与行动',
        year: 2022,
        category: 'agents',
        authors: 'Yao 等',
        venue: 'ICLR 2023',
        url: 'https://arxiv.org/abs/2210.03629',
        pdf: 'https://arxiv.org/pdf/2210.03629',
        summary: '让推理文本与任务动作交替出现，通过外部知识或环境反馈更新后续行动。',
        focus: [
            '跟随一条完整轨迹，区分模型生成的推理、动作和环境返回的观察。',
            '对比只推理、只行动和两者结合的示例，指出反馈在哪一步改变计划。',
            '查看失败轨迹：错误来自推理、动作选择，还是获取到的外部信息？'
        ],
        takeaway: '将生成计划、执行动作和读取反馈串成循环，使后续决策能利用真实的环境观察。',
        limitation: '应连同任务允许的动作与工具接口理解实验；迁移到新环境后要重新检查失败路径。'
    },
    {
        id: 'rag',
        title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
        shortTitle: 'RAG · 检索增强生成',
        year: 2020,
        category: 'agents',
        authors: 'Lewis 等',
        venue: 'NeurIPS 2020',
        url: 'https://arxiv.org/abs/2005.11401',
        pdf: 'https://arxiv.org/pdf/2005.11401',
        summary: '把神经检索器、维基百科向量索引与序列生成模型结合，让生成过程利用外部文档。',
        focus: [
            '画出问题、检索器、文档索引和生成器之间的数据流。',
            'RAG-Sequence 与 RAG-Token 如何使用检索段落，文档选择的粒度有什么不同？',
            '对照检索与生成的实验，区分找到相关文档和生成正确答案这两个环节。'
        ],
        takeaway: '将模型参数中的知识与可检索的外部文档结合，并研究生成过程如何利用多个候选段落。',
        limitation: '检索到文档并不保证答案正确；阅读时还应检查检索覆盖、文档内容和生成结果的对应关系。'
    }
];
