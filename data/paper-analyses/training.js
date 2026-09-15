// Research notes checked against the full papers and authors' implementations.
// Worked examples and application judgments are original teaching material.
Object.assign(window.PAPER_ANALYSES ||= {}, {
    qlora: {
        verdict: 'QLoRA 解决的是“大模型能否装进显存并完成微调”：压缩冻结权重，把学习留给小型适配器。',
        problem: '只减少可训练参数还不够，基础权重、激活和优化器状态仍会占显存。需要分别处理常驻开销与训练峰值。',
        baseline: '对照普通 LoRA 时，要问基础模型以几位存储；对照全量微调时，还要问哪些参数在更新。两种比较回答的问题不同。',
        method: [
            { title: 'NF4 存权重', text: '冻结基础模型并用 4 位 NF4 保存；NF4 的数值分布针对近似正态的权重设计。运算时恢复到 BF16，存储精度与计算精度不同。' },
            { title: '双重量化省元数据', text: '每块权重需要缩放常数，常数本身也会占空间。再量化这些常数，进一步压缩常驻内存。' },
            { title: '分页应对峰值', text: '分页优化器利用 CPU 与 GPU 间的内存迁移管理优化器状态；遇到峰值时有腾挪空间，但迁移并非免费。' },
            { title: '把更新交给 LoRA', text: '反向传播经过冻结模型，更新适配器。作者实现将量化类型、双重量化、计算精度和分页优化器设为独立配置项，读代码时可逐项核对。' }
        ],
        flow: [
            { label: '装载', detail: '基础权重以 NF4 保存，并保存压缩后的缩放信息。' },
            { label: '前向', detail: '按计算需要反量化，与适配器输出相加。' },
            { label: '反向', detail: '由任务损失计算适配器梯度，基础权重保持冻结。' },
            { label: '更新', detail: '优化器更新适配器，分页机制处理内存压力。' }
        ],
        formula: {
            expression: 'Y = X · dequant(W₄) + s · XAB',
            explanation: '简化单层写法：W₄ 是冻结的量化权重；A、B 是可训练低秩矩阵，s 是缩放系数。dequant 恢复用于计算的数值，并不把原权重变成可训练参数。'
        },
        example: {
            title: '教学示例 · 给客服模型加上工单摘要能力',
            text: '设想你有一张显存有限的卡，想让模型把聊天记录整理成“问题、处理、结果”。可以把基础模型理解成一本压缩保存的工具书，适配器像一小册可修改的任务笔记。训练时查阅工具书，用工单数据修改笔记。这里要记两本账：一张记录载入后的固定占用，另一张记录最长工单进入训练时的峰值。若只看模型文件大小，即使成功载入，也可能在反向传播时失败。验证时用同一批未见工单比较漏项、事实错误和峰值显存，再决定是否值得采用；这个例子没有声称实测效果。'
        },
        evidence: {
            intro: '下面分别看“装得下”和“任务分数”，不能把省显存直接换算成加速倍数。',
            items: [
                { label: '微调可行性', value: '65B · 单张 48GB GPU', detail: '论文第 1、4 节报告的配置；全量微调估计需超过 780GB。序列长度、批量及内存管理会影响实际峰值。', url: 'https://arxiv.org/pdf/2305.14314#page=2' },
                { label: '5-shot MMLU 均值', value: '53.1 vs 53.0', detail: '表 4：NF4＋双重量化 vs BF16 LoRA；平均涵盖 LLaMA 7–65B 与 Alpaca、FLAN v2。BF16 行是适配器微调，不能写成 65B 全量微调对照。', url: 'https://arxiv.org/pdf/2305.14314#page=8' }
            ],
            takeaway: '阅读判断：先确认资源是否够用，再检查自己的任务质量。表中均值接近，是选择实验起点的依据，不是所有任务无损的承诺。'
        },
        contributions: [
            '把量化存储、低秩更新与峰值管理组合成可复用的微调方案。',
            '给出不同精度的对照，让显存节省与任务表现可以分别讨论。'
        ],
        limits: [
            '这里证明的是特定配置下的可行性；不能据此推断训练、推理都更快。',
            '应用判断：迁移前应单独检查长输入与领域术语，平均基准分可能掩盖关键失败。'
        ],
        application: {
            fit: '适合先做资源受限的任务适配实验。建议把可用显存、最长输入和质量目标写在同一张实验记录里。',
            caution: '复现时固定数据与评估集，逐项改精度配置；否则无法区分收益来自量化方案还是训练数据。'
        },
        nextReading: [
            { label: '第 3 节 · 精度与更新路径', detail: '沿式 5、6 找出哪部分在存储、哪部分在计算、哪部分会更新。', url: 'https://arxiv.org/pdf/2305.14314#page=5' },
            { label: '表 4 · 大模型精度对照', detail: '先读列头中的规模和数据集，再看均值；继续核对单项变化。', url: 'https://arxiv.org/pdf/2305.14314#page=8' },
            { label: '作者仓库 · 配置入口', detail: '查看 Quantization 与 Paged Optimizer，连接论文概念和实现参数。', url: 'https://github.com/artidoro/qlora#quantization' }
        ],
        sources: [
            { label: 'QLoRA · 论文全文', url: 'https://arxiv.org/pdf/2305.14314' },
            { label: '作者维护的 QLoRA 实现', url: 'https://github.com/artidoro/qlora' }
        ]
    },
    instructgpt: {
        verdict: 'InstructGPT 展示了如何把“会续写”进一步训练成“更愿意按要求完成任务”，关键在反馈目标与数据。',
        problem: '预测文本中下一个词，与满足用户要求并不是同一个目标。模型知道很多，也可能续写问题、忽略格式或编造答案。',
        baseline: '论文同时比较原始 GPT-3、加指令示例的 GPT-3、监督微调 SFT 和 PPO 系列，以区分提示、示范与偏好训练的收益。',
        method: [
            { title: '示范给出起点', text: '标注者写出理想回答，用监督学习微调 GPT-3，得到 SFT 策略。' },
            { title: '排序训练评委', text: '人给同一问题的多个回答排序；奖励模型 RM 学习预测相对偏好，而不是逐词生成标准答案。' },
            { title: 'PPO 改进回答', text: '策略生成回答，RM 给分；PPO 更新策略，同时用相对 SFT 的 KL 惩罚限制偏离。' },
            { title: 'PPO-ptx 保留能力', text: '再混入预训练数据的学习目标，缓解部分 NLP 任务退步。论文默认的 InstructGPT 指 PPO-ptx；普通 PPO 没有这一附加项。' }
        ],
        flow: [
            { label: '示范数据', detail: '问题＋人工回答 → 训练 SFT。' },
            { label: '偏好数据', detail: '同题多个回答＋排序 → 训练 RM。' },
            { label: '策略优化', detail: '新回答 → 奖励与偏离惩罚 → 更新策略。' },
            { label: '独立评估', detail: '未用于训练的用户提示，检查偏好与具体行为。' }
        ],
        formula: {
            expression: '最大化 E[r(x,y) − β log(π(y|x)/πSFT(y|x))] + γ Epretrain[log π(z)]',
            explanation: '第一项争取奖励，第二项约束相对 SFT 的变化；最后一项继续学习预训练文本 z。β、γ 控制权衡。PPO 取 γ＝0，PPO-ptx 保留最后一项。'
        },
        example: {
            title: '教学示例 · “请用两条要点总结会议”',
            text: '假设回答 A 是两条准确要点，B 是一段流畅却漏掉决定的文字，C 虽有两条却编造截止日期。示范阶段给模型看 A；排序阶段让人把 A 排在 B、C 之前，奖励模型学习的是排序关系；策略优化阶段则用学到的评分改进新回答。真正难的是评分准则：若只奖励“两条”，C 也可能得高分。我的阅读建议是额外记录标注理由，把“格式正确、内容忠实、完成任务”分开看，并保留意见不一致的样例，用于检查标注质量。文字理由不是原论文奖励模型的训练目标；这也不是论文中的实验案例。'
        },
        evidence: {
            intro: '这些结果来自论文规定的提示分布与标注流程，衡量的是回答偏好。',
            items: [
                { label: '小模型也更受偏好', value: '1.3B InstructGPT vs 175B GPT-3', detail: '图 1 的 API 提示人工评估支持这一结论；图中纵轴统一以 175B SFT 为比较基线。不能推断所有知识或推理任务都由小模型胜出。', url: 'https://arxiv.org/pdf/2203.02155#page=2' },
                { label: '175B 直接比较', value: '85 ± 3% / 71 ± 4%', detail: '第 4.1 节：分别相对普通 GPT-3、few-shot 提示后的 GPT-3；均为 175B，人工偏好率，误差为 95% 置信区间。', url: 'https://arxiv.org/pdf/2203.02155#page=11' }
            ],
            takeaway: '阅读判断：参数量之外，任务目标也值得投入。做产品评估时应同时记录“用户更喜欢”和“任务确实完成”，避免让一个总分替代全部判断。'
        },
        contributions: [
            '把人工示范、回答排序与策略优化连接到广泛的指令任务。',
            '通过不同训练阶段的对照，展示反馈目标对使用体验的影响。'
        ],
        limits: [
            '模型卡说明反馈来自特定标注者群体，不能代表所有人的价值与偏好。',
            '模型仍可能编造信息、继承偏见或在陌生输入上失常；更听话不等于事实可靠。'
        ],
        application: {
            fit: '适合理解助手训练的完整链路。若已有明确评分准则，可先做小规模盲评，检查人能否稳定区分回答质量。',
            caution: '不要直接把偏好胜率当正确率。对新领域分别设计事实核对、格式检查与人工满意度，并观察不同群体的分歧。'
        },
        nextReading: [
            { label: '第 3.5 节 · 训练目标', detail: '重点读式 2 与 γ＝0 的定义，区分 PPO、PPO-ptx。', url: 'https://arxiv.org/pdf/2203.02155#page=9' },
            { label: '第 4.1 节 · 偏好结果', detail: '看图 3 的提示来源和评委分组，再读直接比较的百分比。', url: 'https://arxiv.org/pdf/2203.02155#page=11' },
            { label: '模型卡 · 方法局限', detail: '阅读标注者选择、语言分布与反馈代表性的讨论。', url: 'https://github.com/openai/following-instructions-human-feedback/blob/main/model-card.md#methodology-limitations' }
        ],
        sources: [
            { label: 'InstructGPT · 论文全文', url: 'https://arxiv.org/pdf/2203.02155' },
            { label: 'OpenAI · 方法介绍', url: 'https://openai.com/index/instruction-following/' },
            { label: 'OpenAI · InstructGPT 模型卡', url: 'https://github.com/openai/following-instructions-human-feedback/blob/main/model-card.md' }
        ]
    },
    dpo: {
        verdict: 'DPO 把成对偏好直接变成语言模型的训练损失，省去独立奖励模型与 PPO 优化循环。',
        problem: '传统 RLHF 需要先学评分函数，再通过反复生成与强化学习优化策略。多个组件使训练和调试更复杂。',
        baseline: '论文对照 PPO、SFT、只训练优选回答的 Preferred-FT，以及从多个候选中挑最高分的 Best-of-N；后者还增加生成成本。',
        method: [
            { title: '准备同题偏好对', text: '每条数据包含问题 x、优选回答 y⁺ 和落选回答 y⁻。作者实现通常先 SFT，再学习偏好。' },
            { title: '保留参考模型', text: 'πref 是固定的参照，通常来自 SFT。比较当前与参考模型的回答概率变化；原始 DPO 仍需参考概率。' },
            { title: '直接优化差值', text: '把奖励与最优策略之间的关系代入偏好模型，同题共有的归一化项抵消，于是得到可直接求梯度的损失。' },
            { title: '离线训练，另行评估', text: '优化时直接计算现有回答的概率，不必像 PPO 持续采样新回答；制作偏好数据和生成式评估仍可能需要采样。' }
        ],
        flow: [
            { label: '建立起点', detail: 'SFT 模型 → 初始化可训练策略与固定参考模型。' },
            { label: '读偏好对', detail: '同一提示下，取优选与落选回答。' },
            { label: '计算概率', detail: '两个模型分别计算两段回答的条件对数概率。' },
            { label: '优化策略', detail: '计算相对差值与分类损失，只更新当前策略。' }
        ],
        formula: {
            expression: 'L = −E log σ(β[log(πθ(y⁺|x)/πref(y⁺|x)) − log(πθ(y⁻|x)/πref(y⁻|x))])',
            explanation: 'σ 把差值映射为偏好概率；β 联系到原目标中的 KL 约束。这里比较的是相对参考模型的变化，不能只看优选回答自身的概率。'
        },
        example: {
            title: '教学示例 · 算一次偏好损失',
            text: '假设 A 是准确的会议摘要，B 编造了决定，标注偏好 A。为便于手算，设参考模型对整段 A、B 的概率为 0.2、0.1，当前模型为 0.3、0.05，并仅在本例取 β＝1。两项比值为 1.5 和 0.5，差值是 log 3，σ(log 3)＝0.75，损失约 0.288；模型与参考完全相同时损失约 0.693。这说明本对数据已朝标注方向移动，但不能说明另一份会议摘要也可靠。如果把偏好标反，公式同样会努力拟合错误方向。以上数值为教学构造，不是论文测量。'
        },
        evidence: {
            intro: '以下都是各模型相对数据集参考摘要的胜率，并非 DPO 与 PPO 两两对决的胜率。',
            items: [
                { label: 'TL;DR 摘要', value: '约 61% vs 57%', detail: '第 6.2 节、图 2：DPO vs PPO，共用 GPT-J SFT 起点，温度均为 0，GPT-4 评判生成摘要与人工参考摘要。', url: 'https://arxiv.org/pdf/2305.18290#page=9' },
                { label: '迁移到 CNN/DailyMail', value: '36% vs 26%', detail: '表 1：TL;DR 训练后的 DPO vs PPO，新闻测试集、温度 0，GPT-4 对照该数据集参考摘要。两者均低于 50%。', url: 'https://arxiv.org/pdf/2305.18290#page=9' }
            ],
            takeaway: '阅读判断：训练流程可以更直接，质量仍要用明确基线衡量。迁移结果说明“优于另一种方法”与“足够好用”是两个需要分别回答的问题。'
        },
        contributions: [
            '把受约束的奖励优化改写成直接训练策略的偏好损失。',
            '提供可运行的 SFT → DPO 实现，方便检查数据、参考概率和更新逻辑。'
        ],
        limits: [
            '论文实验规模最高到 6B；自动评委的提示词会影响胜率，不能直接外推到所有模型与任务。',
            '应用判断：偏好标签若奖励冗长、迎合或错误事实，损失下降也可能偏离实际目标。'
        ],
        application: {
            fit: '已有可信的同题回答对、想做离线偏好训练时，可以先以 DPO 建立对照实验。',
            caution: '先抽查标注理由，再固定一组独立评估。把偏好准确性、事实质量和回答长度分开记录，避免只追训练损失。'
        },
        nextReading: [
            { label: '第 4 节 · 式 4—7', detail: '追踪奖励如何变成策略概率比，以及同题公共项为何消失。', url: 'https://arxiv.org/pdf/2305.18290#page=4' },
            { label: '第 6.2—6.4 节 · 评测边界', detail: '对照温度、参考摘要与 GPT-4 提示，再看人工评价。', url: 'https://arxiv.org/pdf/2305.18290#page=9' },
            { label: '作者实现 · trainers.py', detail: '找到 preference_loss，区分当前策略与参考模型的对数概率。', url: 'https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py' }
        ],
        sources: [
            { label: 'DPO · 论文全文', url: 'https://arxiv.org/pdf/2305.18290' },
            { label: 'DPO · 作者参考实现', url: 'https://github.com/eric-mitchell/direct-preference-optimization' },
            { label: 'DPO · 训练与损失代码', url: 'https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py' }
        ]
    }
});
