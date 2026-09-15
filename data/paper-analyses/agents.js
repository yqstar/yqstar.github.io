// Primary paper versions: ReAct v3; RAG v4. Examples and application judgments are editorial.
Object.assign(window.PAPER_ANALYSES ||= {}, {
    react: {
        verdict: '让模型根据行动获得的新信息改计划，是 ReAct 最值得带走的思想；工具调用次数和解释长度本身都不是能力指标。',
        problem: '只凭记忆推理，第一步认错实体就可能一路推错；只连续执行动作，又容易忘记目标和已经试过的路径。论文研究如何把推理与环境反馈接起来。',
        baseline: 'CoT 只生成推理，Act 只生成动作并读取观察。ReAct 同时保留两者，因此可以分别检查：计划是否帮助选动作，观察是否纠正了计划。',
        method: [
            { title: '用轨迹示范行为', text: '提示中放入人工编写的任务轨迹，包含推理、动作及真实观察。主实验冻结 PaLM-540B；这是一种少样本提示方法，论文另有小模型微调实验。' },
            { title: '把动作交给环境执行', text: '问答接口提供 search、lookup、finish。作者代码中，search 读取词条或返回相近标题，lookup 查找当前词条的匹配句；模型写出的动作必须经过接口才会产生观察。' },
            { title: '依据反馈更新状态', text: '用简短推理记录已知信息、未完成目标和下一步。问答轨迹密集交替，长行动任务可稀疏插入推理；不是每次点击都必须先写一段解释。' }
        ],
        flow: [
            { label: '明确缺口', detail: '当前还缺哪条信息，或哪个子目标尚未完成？' },
            { label: '选择动作', detail: '生成工具允许的动作及参数。' },
            { label: '接收观察', detail: '读取环境返回值，不把模型猜测当作执行结果。' },
            { label: '修正或结束', detail: '新证据支持继续、换路，还是给出答案？' }
        ],
        formula: {
            expression: '上下文 → 推理 / 动作 → 外部观察 → 更新上下文 ↺',
            explanation: '这是流程示意，不是训练损失。推理文本只更新上下文；执行外部动作才会改变环境或取得新信息。后续动作必须基于实际返回值。'
        },
        example: {
            title: '教学示例：找到一本书的存放位置',
            text: '假设任务是“找出《星海手册》在哪个书架”。先查书名，目录返回两个同名版本；这时直接报书架会把检索命中误当成任务完成。下一步根据用户给的作者筛选，再查馆藏位置。若返回“已外借”，结论应保留该状态，不能把“应在 A3 架”说成“现已找到”。这个自拟例子中，真正起作用的是歧义被发现后改变了动作，而非模型写出了多长的计划。'
        },
        evidence: {
            intro: '以下均为论文主表的 PaLM-540B 提示实验；不同任务的指标和示例数不同。',
            items: [
                { label: 'HotpotQA · 答案完全匹配', value: '27.4 vs 29.4', detail: '表 1：ReAct EM 27.4，CoT 29.4，Act 25.7；只给问题，ReAct 用 6 条示范并可查维基百科。ReAct 并未在此全面胜过 CoT。', url: 'https://arxiv.org/pdf/2210.03629v3#page=5' },
                { label: 'ALFWorld · 成功率', value: '71%（最佳）/ 57%（均值）', detail: '表 3：134 个未见游戏、每类任务 2 条示范。71% 是 6 组提示的最佳值；Act 最佳 45%，BUTLER 为 8 次试验最佳 37%，不可把 71% 当平均值。', url: 'https://arxiv.org/pdf/2210.03629v3#page=8' },
                { label: 'WebShop · 满足全部要求', value: '40.0% vs 30.1%', detail: '表 4：500 条测试指令、1 条示范，ReAct 与 Act 成功率分别为 40.0% 和 30.1%；专家为 59.6%。这是模拟购物任务。', url: 'https://arxiv.org/pdf/2210.03629v3#page=8' }
            ],
            takeaway: '本站解读：收益取决于任务是否需要外部反馈，以及模型能否据此转向。成功率比较要同时看模型、示范、工具和试验选择方式。'
        },
        contributions: [
            '把自由文本推理与可执行动作放入同一条轨迹，提供一种可以逐步检查的交互方式。',
            '通过 CoT、Act 等对照拆开“会推理”和“会取信息”的作用，提示两种能力可以互补。'
        ],
        limits: [
            '论文观察到重复动作和无用搜索会导致失败；加入工具并不能保证信息相关，也不能保证读懂返回值。',
            '本站判断：流畅的推理文本不是内部决策忠实性的证明。主表提示结果也不能直接代表任意小模型或现代代理框架。'
        ],
        application: {
            fit: '本站建议：适合查询结果会改变下一步的资料核对、排障、界面任务。先用同一批任务比较成功率、调用成本和失败后的恢复能力。',
            caution: '先定义完成条件和失败出口；连续无新信息时停止重试。若一次查询就足够，多轮循环可能只增加等待时间。'
        },
        nextReading: [
            { label: '图 1 · 完整轨迹', detail: '逐步比较同题的 CoT、Act 与 ReAct，定位反馈改变决策的位置。', url: 'https://arxiv.org/pdf/2210.03629v3#page=2' },
            { label: '§3.3 / 表 2 · 失败分析', detail: '看搜索失败、重复步骤与推理错误如何分开统计。', url: 'https://arxiv.org/pdf/2210.03629v3#page=6' },
            { label: '§4 / 表 3、4 · 行动实验', detail: '连同最佳试验、平均值及示范数量一起读。', url: 'https://arxiv.org/pdf/2210.03629v3#page=8' }
        ],
        sources: [
            { label: '原论文 · ICLR 2023 / v3', url: 'https://arxiv.org/abs/2210.03629v3' },
            { label: '作者项目页 · 提示与微调示例', url: 'https://react-lm.github.io/' },
            { label: '作者代码 · WikiEnv 动作实现', url: 'https://github.com/ysymyth/ReAct/blob/master/wikienv.py' }
        ]
    },
    rag: {
        verdict: '这篇 RAG 研究的是如何训练检索器与生成器，并对候选文档的生成概率求和；它比“搜几段文字塞进提示”更具体。',
        problem: '模型记住了知识，也可能提取不准或无法随资料更新。论文希望让答案同时利用模型参数和可替换的外部文档，而不把全部知识都压进权重。',
        baseline: '闭卷生成只用参数知识；检索后抽取则要求从材料中截取答案。RAG 检索相关段落后生成答案，并允许多个候选段落共同影响输出概率。',
        method: [
            { title: 'DPR 找段落，BART 生成', text: '查询编码器把问题变成向量，在文档索引中找 top-K 段落。每个段落分别与问题组合后交给 BART；不是先把所有段落拼成一份长上下文。' },
            { title: '区别在求和的位置', text: 'Sequence 对每篇候选文档计算整段答案概率，再加权求和；Token 对每个输出位置先汇总各文档的词概率，再组合整段答案概率。' },
            { title: '候选集固定，联合学习', text: '同一输入的 top-K 在生成期间固定，Token 不会逐词发起新检索。训练更新查询编码器和 BART；文档编码器及索引冻结，避免反复重建索引。' }
        ],
        flow: [
            { label: '编码问题', detail: '得到与文档向量可比较的查询表示。' },
            { label: '检索 K 段', detail: '为候选段落计算检索概率。' },
            { label: '分别生成', detail: '在每个候选段落条件下计算词或序列概率。' },
            { label: '汇总概率', detail: '按 Sequence 或 Token 的方式对文档求和。' }
        ],
        formula: {
            expression: 'Sequence：Σ_z a_z ∏ᵢ b_zi\nToken：∏ᵢ Σ_z a_z b_zi',
            explanation: '两式均为 top-K 近似的 p(y|x)。a_z 是文档 z 的检索概率；b_zi 是给定问题、文档和已生成前缀时，第 i 个词的概率。Σ_z 遍历同一候选集；文档是隐变量，求和不等于硬选一篇。'
        },
        example: {
            title: '教学示例：两篇材料怎样影响答案',
            text: '设一个虚构展览的资料 A 写“周二开幕”，资料 B 写“周五闭幕”，两篇都被检索到。Sequence 分别评估“周二至周五”这一整段答案在 A、B 条件下的概率，再加权；Token 则可以在生成“周二”与“周五”时获得不同文档的更强支持。两种形式都可能结合模型已有知识，但这种计算并不保证组合正确。若 B 实际讲的是上一届展览，仍会出现有来源却答错的问题。这是自拟示例，不是论文实验。'
        },
        evidence: {
            intro: '原实验使用 2018 年 12 月维基百科，约 2100 万个 100 词段落；下列为表 1 测试集 EM（完全匹配）。',
            items: [
                { label: 'Natural Questions', value: '44.5 vs 41.5', detail: 'RAG-Sequence 44.5，DPR 检索抽取系统 41.5；RAG-Token 为 44.1。比较的是完整问答系统，不是单独检索召回率。', url: 'https://arxiv.org/pdf/2005.11401v4#page=6' },
                { label: 'TriviaQA · 标准开放域划分', value: '56.8 vs 57.9', detail: 'RAG-Sequence 56.8，DPR 57.9。另一套 TQA-Wiki 划分上 RAG-Sequence 为 68.0，不能把两个划分的数值混在一起宣布领先。', url: 'https://arxiv.org/pdf/2005.11401v4#page=6' }
            ],
            takeaway: '本站解读：这些结果支持检索与生成结合的价值，同时说明效果依赖任务和评估划分，不能推出“只要接知识库就会更准确”。'
        },
        contributions: [
            '把预训练检索器与生成器连接起来，用输入—答案对学习任务所需的检索行为。',
            '给出序列级和词级两种文档隐变量建模，明确“多个来源如何影响生成”的概率形式。'
        ],
        limits: [
            'RAG 微调不要求文档标注，但初始化的 DPR 已使用问答检索监督，不能称为整个流程完全无检索监督。',
            '本站判断：文档相关、答案正确、引用忠实是三个问题。原论文结果不直接覆盖多模态知识库、权限过滤或任意现代聊天模型。'
        ],
        application: {
            fit: '本站建议：将它作为知识库问答的分析框架。像官方实现一样，把检索评估和端到端答案评估分开；先判断丢失发生在找材料还是使用材料。',
            caution: '拿少量自有问答分别测试：有正确文档、缺少文档、文档相互冲突。仅提高命中率，未必解决过期内容或错误拼接。'
        },
        nextReading: [
            { label: '§2.1–2.4 · 方法与训练', detail: '重点比较求和与连乘的次序，以及哪些参数被冻结。', url: 'https://arxiv.org/pdf/2005.11401v4#page=3' },
            { label: '图 2 · 文档后验', detail: '观察生成不同词时，各候选文档的支持如何改变。', url: 'https://arxiv.org/pdf/2005.11401v4#page=7' },
            { label: '§4.5 / 表 6 · 检索消融', detail: '比较冻结检索器、BM25 与联合训练；该表为开发集。', url: 'https://arxiv.org/pdf/2005.11401v4#page=8' }
        ],
        sources: [
            { label: '原论文 · NeurIPS 2020 / v4', url: 'https://arxiv.org/abs/2005.11401v4' },
            { label: '官方实现 · 固定版本 README', url: 'https://github.com/huggingface/transformers/blob/v4.11.3/examples/research_projects/rag/README.md' }
        ]
    }
});
