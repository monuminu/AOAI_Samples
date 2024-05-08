Large language models (LLMs) such as LLaMA and OpenAI’s GPT-4 are revolutionizing technology. However, one of the common complaints about LLMs is their speed, or lack thereof. In many cases, it takes a long time to get an answer from them. This limits LLMs’ applications and their usefulness in latency-critical functions, such as chatbots, copilots, and industrial controllers.

PUBLICATION
Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding 
To address this question, researchers from Microsoft Research and Tsinghua University proposed Skeleton-of-Thought (SoT), a new approach to accelerate generation of LLMs. Unlike most prior methods, which require modifications on the LLM models, systems, or hardware, SoT treats LLMs as black boxes and can therefore be applied on any off-the-shelf open-source (e.g., LLaMA) or even API-based (e.g., OpenAI’s GPT-4) models. Our evaluation shows that not only does SoT considerably accelerate content generation among the 12 LLMs examined, it may also improve the answer quality in some cases. For example, on OpenAI’s GPT-3.5 and GPT-4, SoT provides 2x speed-up while improving the answer quality on benchmark datasets.


![image](https://github.com/microsoft/AsiaAIGBB/assets/33065876/0b094ff6-e7c9-49c1-89b6-5353921f4b5c)

SoT: Encouraging structured thinking in LLMs
The idea of SoT stems from the difference in how LLMs and humans process information. LLMs generate answers sequentially. For example, to answer “How can I improve my time management techniques?” in Figure 1 (left), the LLM finishes one point before moving to the next. In contrast, humans may not always think about questions and write answers sequentially. In many cases, humans first derive the skeleton of the answer and then add details to explain each point. For example, to answer the same question in Figure 1, a person might first think about a list of relevant time management techniques before digging into the details of each technique. This is especially the case for exercises like offering consultancy, taking tests, writing papers, and so on. 

Can we make LLMs process information more dynamically and less linearly? As illustrated in Figure 2, SoT does the trick. Instead of generating answers sequentially, SoT decomposes the generation into two stages: (1) SoT first asks the LLM to derive a skeleton of the answer, and then (2) asks the LLM to provide the answer to each point in the skeleton. This method offers a new opportunity for acceleration, as the answers to separate points in stage 2 can be generated in parallel. This can be done for both local models, whose weights are accessible by the users (e.g., LLaMA), and API-based models which can only be accessed through APIs (e.g., OpenAI’s GPT-4).

For API-based models, we can issue parallel API requests, one for each point. 
For models that are running locally, we can answer all points simultaneously in a batch. Note that in many scenarios (e.g., local service, centralized service within unsaturated query period), the decoding phase of LLMs is usually bottlenecked by weight loading instead of activation loading or computation, and thus underutilizes available hardware. In these cases, running LLM inference with increased batch sizes improves the hardware computation utilization and does not significantly increase latency.
Consequently, if there are B points in the answer, generating these points in parallel as in SoT can theoretically give up to Bx speed-up compared to sequential generation as in current LLMs. However, in practice, due to the extra skeleton stage, unbalanced point lengths, and other overheads, the actual speed-up can be smaller.

![image](https://github.com/microsoft/AsiaAIGBB/assets/33065876/b6f6ddbe-86b3-4415-9e52-abd127b510b6)
