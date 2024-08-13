<script>
	import tailwindConfig from '../../../tailwind.config';
	import resolveConfig from 'tailwindcss/resolveConfig';
	// import { base } from '$app/paths';

	// import Youtube from './Youtube.svelte';

	let softmaxEquation = `$$\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}$$`;
	let reluEquation = `$$\\text{ReLU}(x) = \\max(0,x)$$`;

	let currentPlayer;

	const { theme } = resolveConfig(tailwindConfig);
</script>

<div id="description">
	<div class="article-section">
		<h1>什么是Transformer?</h1>

		<p>
			Transformer 是一种神经网络架构，它从根本上改变了人工智能的方法。Transformer 首次出现于 2017 年的开创性论文
			<a
				href="https://dl.acm.org/doi/10.5555/3295222.3295349"
				title="ACM Digital Library"
				target="_blank">"Attention is All You Need"</a
			>
			中，此后成为深度学习模型的首选架构，为 OpenAI 的 <strong>GPT</strong>、Meta 的 <strong>Llama</strong> 和 Google 的<strong>Gemini</strong> 
			等文本生成模型提供支持。除了文本之外，Transformer 还被应用于
			<a
				href="https://huggingface.co/learn/audio-course/en/chapter3/introduction"
				title="Hugging Face"
				target="_blank">音频生成</a
			>,
			<a
				href="https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification"
				title="Hugging Face"
				target="_blank">图像识别</a
			>,
			<a href="https://elifesciences.org/articles/82819" title="eLife"
				>蛋白质结构预测</a
			>, 甚至
			<a
				href="https://www.deeplearning.ai/the-batch/reinforcement-learning-plus-transformers-equals-efficiency/"
				title="Deep Learning AI"
				target="_blank">游戏</a
			>, 证明了其在众多领域的广泛适用性。
		</p>
		<p>
			从根本上说，文本生成 Transformer 模型基于下一个词预测的原则运作：给定用户的文本提示，最有可能出现在该输入之后的下一个词是什么？
			Transformer 的核心创新和强大之处在于其对自注意力机制的使用，这使得它们能够比以前的架构更有效地处理整个序列并捕获远程依赖关系。
		</p>
		<p>
			<a href="https://huggingface.co/openai-community/gpt2" title="Hugging Face" target="_blank"
			>GPT-2</a
		>模型系列是文本生成 Transformer 的杰出代表。Transformer Explainer 由具有 1.24 亿个参数的 GPT-2（小型）模型提供支持。虽然它不是最新或最强大的 Transformer 模型，
			但它与当前最先进的模型共享许多相同的架构组件和原理，使其成为理解基础知识的理想起点。
		</p>
	</div>

	<div class="article-section">
		<h1>Transformer 架构</h1>

		<p>
			每一个文本生成式 Transformer 包含如下 <strong>三个关键部分</strong>:
		</p>
		<ol>
			<li>
				<strong class="bold-purple">Embedding</strong>: 文本输入被分成称为“token”的更小的单元，
				可以是单词或子词。这些标记被转换为称为“Embedding”的数字向量，它们捕捉单词的语义。
			</li>
			<li>
				<strong class="bold-purple">Transformer 块</strong>是模型的基本构建块，用于处理和转换输入数据。每个块包括：
				<ul class="">
				  <li>
					<strong>注意力机制</strong>是 Transformer 块的核心组件。它允许词元与其他词元进行通信，捕获上下文信息和词语之间的关系。
				  </li>
				  <li>
					<strong>MLP（多层感知机）层</strong>是一个前馈神经网络，独立地对每个词元进行操作。注意力层的目标是在词元之间路由信息，而 MLP 的目标是完善每个词元的表示。
				  </li>
				</ul>
			  </li>
			  <li>
				<strong class="bold-purple">输出概率</strong>：最终的线性层和 softmax 层将处理后的嵌入转换为概率，使模型能够对序列中的下一个词元进行预测。
			  </li>
		</ol>

		<div class="architecture-section">
			<h2>Embedding</h2>
			<p>
				假设你想使用 Transformer 模型生成文本。 你添加了像这样的提示：<code>“Data visualization empowers users to”</code>。 此输入需要转换为模型可以理解和处理的格式。 这就是嵌入的用武之地：它将文本转换为模型可以处理的数字表示。 要将提示转换为嵌入，我们需要 1) 对输入进行标记化，2) 获取词元嵌入，3) 添加位置信息，最后 4) 将词元和位置编码相加得到最终嵌入。 让我们看看这些步骤是如何完成的。
			  </p>
			  <div class="figure">
				<img src="./article_assets/embedding.png" alt="embedding" width="60%" height="60%" align="middle" />
			  </div>
			  <div class="figure-caption">
				图 <span class="attention">1</span>。 展开“嵌入”层视图，显示输入提示如何转换为向量表示。 该过程涉及
				<span class="fig-numbering">(1)</span> 标记化、(2) 词元嵌入、(3) 位置编码和 (4) 最终嵌入。
			  </div>
			  <div class="article-subsection">
				<h3>步骤 1：标记化</h3>
				<p>
				  标记化是将输入文本分解成更小、更易于管理的部分（称为词元）的过程。 这些词元可以是一个词或一个子词。 单词 <code
					>"Data"</code
				  >
				  和 <code>"vizualization"</code> 对应于唯一的词元，而单词 <code>"empowers"</code>
				  被拆分为两个词元。 词元的完整词汇表是在训练模型之前决定的：GPT-2 的词汇表有 <code>50,257</code> 个唯一的词元。 现在我们已经将输入文本拆分为具有不同 ID 的词元，我们可以从嵌入中获取它们的向量表示。
				</p>
			</div>
			<div class="article-subsection">
				<h3>步骤 2. 词元嵌入</h3>
				<p>
				 GPT-2 Small 将词汇表中的每个词元表示为一个 768 维的向量；向量的维度取决于模型。 这些嵌入向量存储在一个形状为 <code>(50,257, 768)</code> 的矩阵中，包含大约 3900 万个参数！ 这个庞大的矩阵允许模型为每个词元分配语义。
				</p>
			  </div>
			  <div class="article-subsection">
				<h3>步骤 3. 位置编码</h3>
				<p>
				 嵌入层还对每个词元在输入提示中的位置信息进行编码。 不同的模型使用不同的位置编码方法。 GPT-2 从头开始训练自己的位置编码矩阵，并将其直接集成到训练过程中。
				</p>
			  </div>
			  <div class="article-subsection">
				<h3>步骤 4. 最终嵌入</h3>
				<p>
				 最后，我们将词元编码和位置编码相加，得到最终的嵌入表示。 这种组合表示既捕获了词元的语义，也捕获了它们在输入序列中的位置。
				</p>
			  </div>
		</div>

		<div class="architecture-section">
			<h2>Transformer Block</h2>

			<p>
				Transformer 处理的核心在于 Transformer 块，它由多头自注意力机制和多层感知机层组成。
				大多数模型由多个这样的块组成，这些块按顺序依次堆叠。
				词元表示从第一个块到第十二个块逐层演化，使模型能够对每个词元建立起复杂的理解。
				这种分层方法导致了对输入的更高阶表示。
			</p>

			<div class="article-subsection">
				<h3>多头自注意力机制 Multi-Head Self-Attention</h3>
				<p>
					自注意力机制使模型能够专注于输入序列的相关部分，从而使其能够捕获数据中的复杂关系和依赖关系。
					让我们逐步了解如何计算这种自注意力。
				</p>
				<div class="article-subsection-l2">
					<h4>第一步: Query, Key, and Value Matrices</h4>

					<div class="figure">
						<img src="./article_assets/QKV.png" alt="QKV" width="80%" align="middle" />
					</div>
					<div class="figure-caption">
						Figure <span class="attention">2</span>. Computing Query, Key, and Value matrices from
						the original embedding.
					</div>

					<p>
						Each token's embedding vector is transformed into three vectors:
						<span class="q-color">Query (Q)</span>,
						<span class="k-color">Key (K)</span>, and
						<span class="v-color">Value (V)</span>. These vectors are derived by multiplying the
						input embedding matrix with learned weight matrices for
						<span class="q-color">Q</span>,
						<span class="k-color">K</span>, and
						<span class="v-color">V</span>. Here's a web search analogy to help us build some
						intuition behind these matrices:
					</p>
					<ul>
						<li>
							<strong class="q-color font-medium">Query (Q)</strong> is the search text you type in
							the search engine bar. This is the token you want to
							<em>"find more information about"</em>.
						</li>
						<li>
							<strong class="k-color font-medium">Key (K)</strong> is the title of each web page in the
							search result window. It represents the possible tokens the query can attend to.
						</li>
						<li>
							<strong class="v-color font-medium">Value (V)</strong> is the actual content of web pages
							shown. Once we matched the appropriate search term (Query) with the relevant results (Key),
							we want to get the content (Value) of the most relevant pages.
						</li>
					</ul>
					<p>
						By using these QKV values, the model can calculate attention scores, which determine how
						much focus each token should receive when generating predictions.
					</p>
				</div>
				<div class="article-subsection-l2">
					<h4>第二步: 掩码自注意力</h4>
					<p>
						Masked 自注意力机制通过专注于输入的相关部分，同时阻止访问未来的词元，从而使模型能够生成序列。
					</p>

					<div class="figure">
						<img src="./article_assets/attention.png" alt="attention" width="80%" align="middle" />
					</div>
					<div class="figure-caption">
						Figure <span class="attention">3</span>. Using Query, Key, and Value matrices to
						calculate masked self-attention.
					</div>

					<ul>
						<li>
							<strong>注意力分数 Attention Score</strong>：<span class="q-color">Query</span>
							和 <span class="k-color">Key</span> 矩阵的点积决定了每个查询与每个键的对齐，生成一个反映所有输入词元之间关系的方阵。
						</li>
						<li>
							<strong>掩码</strong>：对注意力矩阵的上三角应用掩码，以防止模型访问未来的词元，将这些值设置为负无穷大。 
							模型需要学习如何在不“窥视”未来的情况下预测下一个词元。
						</li>
						<li>
							<strong>Softmax</strong>：掩码后，注意力分数通过 softmax 操作转换为概率，该操作取每个注意力分数的指数。 
							矩阵的每一行总和为 1，表示该行词元左侧所有其他词元的相关性。
						</li>
					</ul>
				</div>
				<div class="article-subsection-l2">
					<h4>第三步: 输出</h4>
					<p>
						The model uses the masked self-attention scores and multiplies them with the
						<span class="v-color">Value</span> matrix to get the
						<span class="purple-color">final output</span>
						of the self-attention mechanism. GPT-2 has <code>12</code> self-attention heads, each capturing
						different relationships between tokens. The outputs of these heads are concatenated and passed
						through a linear projection.
					</p>
				</div>

				<div class="article-subsection">
					<h3>MLP: Multi-Layer Perceptron</h3>

					<div class="figure">
						<img src="./article_assets/mlp.png" alt="mlp" width="70%" align="middle" />
					</div>
					<div class="figure-caption">
						Figure <span class="attention">4</span>. Using MLP layer to project the self-attention
						representations into higher dimensions to enhance the model's representational capacity.
					</div>

					<p>
						在多头自注意力机制捕获了输入词元之间的各种关系之后，连接后的输出会传递给多层感知机 (MLP) 层，以增强模型的表示能力。
						MLP 块由两个线性变换组成，中间有一个 GELU 激活函数。第一个线性变换将输入的维度增加了四倍，
						从 <code>768</code> 增加到 <code>3072</code>。第二个线性变换将维度减少回原来的大小
						 <code>768</code>，确保后续层接收维度一致的输入。与自注意力机制不同，MLP 独立处理词元，
						 只是简单地将它们从一种表示映射到另一种表示。
					</p>
				</div>

				<div class="architecture-section">
					<h2>输出概率</h2>
					<p>
						在输入经过所有 Transformer 块处理之后，输出会被传递到最终的线性层，以便为词元预测做好准备。 该层将最终表示投影到一个 <code>50,257</code> 维的空间中，词汇表中的每个词元在这个空间中都有一个对应的值，称为 <code>logit</code>。 任何词元都可以是下一个词，因此此过程允许我们简单地根据词元成为下一个词的可能性对它们进行排序。 然后，我们应用 softmax 函数将 logits 转换为总和为 1 的概率分布。 这将允许我们根据词元的可能性对下一个词元进行采样。
					  </p>
					  
					  <div class="figure">
						<img src="./article_assets/softmax.png" alt="softmax" width="60%" align="middle" />
					  </div>
					  <div class="figure-caption">
						图 <span class="attention">5</span>。 词汇表中的每个词元都会根据模型的输出 logits 被分配一个概率。 这些概率决定了每个词元成为序列中下一个词的可能性。
					  </div>
					  
					  <p>
						最后一步是通过从这个分布中采样来生成下一个词元。<code>temperature</code>
						超参数在这个过程中起着至关重要的作用。 从数学上讲，这是一个非常简单的操作：模型输出 logits 只是简单地除以 <code>temperature</code>：
					  </p>
					  
					  <ul>
						<li>
						  <code>temperature = 1</code>：将 logits 除以 1 对 softmax 输出没有影响。
						</li>
						<li>
						  <code>temperature &lt; 1</code>：较低的温度通过锐化概率分布使模型更加自信和确定性，从而导致更可预测的输出。
						</li>
						<li>
						  <code>temperature &gt; 1</code>：较高的温度会创建更柔和的概率分布，从而在生成的文本中允许更多的随机性——有些人将其称为模型的“<em>创造力</em>”。
						</li>
					  </ul>
					  
					  <p>
						调整温度，看看您如何在确定性和多样化输出之间取得平衡！
					  </p>
				</div>

				<div class="architecture-section">
					<h2>高级架构特性</h2>

					<p>
						有几种先进的架构特性可以增强 Transformer 模型的性能。 
						虽然这些特性对模型的整体性能很重要，但对于理解架构的核心概念来说并不那么重要。 
						层归一化、Dropout 和残差连接是 Transformer 模型中的关键组件，特别是在训练阶段。 
						层归一化可以稳定训练并帮助模型更快地收敛。 Dropout 通过随机停用神经元来防止过拟合。 
						残差连接允许梯度直接流经网络，并有助于防止梯度消失问题。
					</p>
					<div class="article-subsection">
						<h3>Layer Normalization</h3>

						<p>
							层归一化有助于稳定训练过程并改善收敛性。 它通过跨特征归一化输入来工作，确保激活值的均值和方差一致。 
							这种归一化有助于缓解与内部协变量偏移相关的问题，使模型能够更有效地学习并降低对初始权重的敏感性。 
							在每个 Transformer 块中应用两次层归一化，一次在自注意力机制之前，一次在 MLP 层之前。
						</p>
					</div>
					<div class="article-subsection">
						<h3>Dropout</h3>

						<p>
							Dropout 是一种正则化技术，用于通过在训练期间将一部分模型权重随机设置为零来防止神经网络过拟合。
							这鼓励模型学习更鲁棒的特征，并减少对特定神经元的依赖，帮助网络更好地泛化到新的、未见过的数据。
							在模型推理期间，Dropout 被停用。这实质上意味着我们正在使用训练好的子网络的集合，这会带来更好的模型性能。
						</p>
					</div>
					<div class="article-subsection">
						<h3>残差连接 Residual connections</h3>

						<p>
							残差连接最初是在 2015 年的 ResNet 模型中引入的。这种架构创新通过实现对非常深层的神经网络的训练，
							彻底改变了深度学习。本质上，残差连接是绕过一层或多层的捷径，将一层的输入添加到其输出中。
							这有助于缓解梯度消失问题，使得更容易训练具有多个 Transformer 块彼此堆叠的深度网络。
							在 GPT-2 中，每个 Transformer 块中使用两次残差连接：一次在 MLP 之前，一次在 MLP 之后，
							确保梯度更容易流动，并且早期层在反向传播期间接收到足够的更新。
						</p>
					</div>
				</div>

				<div class="article-section">
					<h1>交互功能</h1>
					<p>
						Transformer Explainer 旨在实现交互性，并允许您探索 Transformer 的内部工作原理。 以下是一些您可以使用的交互功能：
					</p>

					<ul>
						<li>
							<strong>输入您自己的文本序列</strong>，以查看模型如何处理它并预测下一个词。 探索注意力权重、中间计算结果，并查看如何计算最终输出概率。
						</li>
						<li>
							<strong>使用温度滑块</strong>来控制模型预测的随机性。 探索如何通过更改温度值来使模型输出更具确定性或更具创造性。
						</li>
						<li>
							<strong>与注意力图互动</strong>，以查看模型如何关注输入序列中的不同词元。 将鼠标悬停在词元上以突出显示其注意力权重，并探索模型如何捕获上下文和词语之间的关系。
						</li>
					</ul>
				</div>

				<div class="article-section">
					<h2>Transformer Explainer 如何实现？</h2>
					<p>
						Transformer Explainer 的特色在于直接在浏览器中运行的 GPT-2（小型）实时模型。 该模型源自 Andrej Karpathy 的
						<a href="https://github.com/karpathy/nanoGPT" title="Github" target="_blank"
							>nanoGPT 项目</a
						>
						中的 PyTorch GPT 实现，并已转换为
						<a href="https://onnxruntime.ai/" title="ONNX" target="_blank">ONNX Runtime</a
						>
						，以实现无缝的浏览器内执行。 该界面使用 JavaScript 构建，
						<a href="https://kit.svelte.dev/" title="Svelte" target="_blank">Svelte</a
						>
						作为前端框架，
						<a href="http://D3.js" title="D3" target="_blank">D3.js</a
						>
						用于创建动态可视化。 数值会根据用户输入实时更新。
					</p>
				</div>

				<div class="article-section">
					<h2>谁开发了本项目?</h2>
					<p>
						<a href="https://aereeeee.github.io/" target="_blank">Aeree Cho</a>,
						<a href="https://www.linkedin.com/in/chaeyeonggracekim/" target="_blank">Grace C. Kim</a
						>,
						<a href="https://alexkarpekov.com/" target="_blank">Alexander Karpekov</a>,
						<a href="https://alechelbling.com/" target="_blank">Alec Helbling</a>,
						<a href="https://zijie.wang/" target="_blank">Jay Wang</a>,
						<a href="https://seongmin.xyz/" target="_blank">Seongmin Lee</a>,
						<a href="https://bhoov.com/" target="_blank">Benjamin Hoover</a>, and
						<a href="https://poloclub.github.io/polochau/" target="_blank">Polo Chau</a>
						<a href="https://github.com/ZhangWei-KUMO" target="_blank">中文翻译：Lewis Zhang</a>
					</p>
				</div>
			</div>
		</div>
	</div>
</div>

<style lang="scss">
	a {
		color: theme('colors.blue.500');

		&:hover {
			color: theme('colors.blue.700');
		}
	}

	.bold-purple {
		color: theme('colors.purple.700');
		font-weight: bold;
	}

	code {
		color: theme('colors.gray.500');
		background-color: theme('colors.gray.50');
		font-family: theme('fontFamily.mono');
	}

	.q-color {
		color: theme('colors.blue.400');
	}

	.k-color {
		color: theme('colors.red.400');
	}

	.v-color {
		color: theme('colors.green.400');
	}

	.purple-color {
		color: theme('colors.purple.500');
	}

	.article-section {
		padding-bottom: 2rem;
	}
	.architecture-section {
		padding-top: 1rem;
	}
	.video-container {
		position: relative;
		padding-bottom: 56.25%; /* 16:9 aspect ratio */
		height: 0;
		overflow: hidden;
		max-width: 100%;
		background: #000;
	}

	.video-container iframe {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
	}

	#description {
		padding-bottom: 3rem;
		margin-left: auto;
		margin-right: auto;
		max-width: 78ch;
	}

	#description h1 {
		color: theme('colors.purple.700');
		font-size: 2.2rem;
		font-weight: 300;
		padding-top: 1rem;
	}

	#description h2 {
		// color: #444;
		color: theme('colors.purple.700');
		font-size: 2rem;
		font-weight: 300;
		padding-top: 1rem;
	}

	#description h3 {
		color: theme('colors.gray.700');
		font-size: 1.6rem;
		font-weight: 200;
		padding-top: 1rem;
	}

	#description h4 {
		color: theme('colors.gray.700');
		font-size: 1.6rem;
		font-weight: 200;
		padding-top: 1rem;
	}

	#description p {
		margin: 1rem 0;
	}

	#description p img {
		vertical-align: middle;
	}

	#description .figure-caption {
		font-size: 0.8rem;
		margin-top: 0.5rem;
		text-align: center;
		margin-bottom: 2rem;
	}

	#description ol {
		margin-left: 3rem;
		list-style-type: decimal;
	}

	#description li {
		margin: 0.6rem 0;
	}

	#description p,
	#description div,
	#description li {
		color: theme('colors.gray.600');
		// font-size: 17px;
		// font-size: 15px;
		line-height: 1.6;
	}

	#description small {
		font-size: 0.8rem;
	}

	#description ol li img {
		vertical-align: middle;
	}

	#description .video-link {
		color: theme('colors.blue.600');
		cursor: pointer;
		font-weight: normal;
		text-decoration: none;
	}

	#description ul {
		list-style-type: disc;
		margin-left: 2.5rem;
		margin-bottom: 1rem;
	}

	#description a:hover,
	#description .video-link:hover {
		text-decoration: underline;
	}

	.figure,
	.video {
		width: 100%;
		display: flex;
		flex-direction: column;
		align-items: center;
	}
</style>
