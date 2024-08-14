# Transformer 解释器: 文本生成式模式交互学习网站

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![arxiv badge](https://img.shields.io/badge/arXiv-2408.04619-red)](https://arxiv.org/abs/2408.04619)

Transformer Expander 是一种交互式可视化工具，旨在帮助任何人了解 GPT 等基于 Transformer 的模型如何工作。它直接在您的浏览器中运行实时 GPT-2 模型，允许您试验自己的文本并实时观察 Transformer 的内部组件和操作如何协同工作以预测下一个令牌。
<table>
<tr>
    <td colspan="2"><video width="100%" src='https://github.com/poloclub/transformer-explainer/assets/5067740/5c2d6a9d-2cbf-4b01-9ce1-bdf8e190dc42'></td>
</tr>

</table>


### 研究论文
[**Transformer Explainer: Interactive Learning of Text-Generative Models**](https://arxiv.org/abs/2408.04619).
Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Zijie J. Wang, Seongmin Lee, Benjamin Hoover, Duen Horng Chau.
*Poster, IEEE VIS 2024.*

## How to run locally

#### 运行环境

- Node.js 20 or higher
- NPM

#### CloudFlare一键部署

本项目支持CloudFlare Pages一键部署。由于本项目的模型下载源为HuggingFace对于中国用户而已需要在使用时开启科学上网。

#### 步骤
## 测试
```bash
git clone https://github.com/poloclub/transformer-explainer.git
cd transformer-explainer
npm install
npm run dev
```

之后打开浏览器 http://localhost:5173.

## Credits

Transformer Explainer was created by <a href="https://aereeeee.github.io/" target="_blank">Aeree Cho</a>, <a href="https://www.linkedin.com/in/chaeyeonggracekim/" target="_blank">Grace C. Kim</a>, <a href="https://alexkarpekov.com/" target="_blank">Alexander Karpekov</a>, <a href="https://alechelbling.com/" target="_blank">Alec Helbling</a>, <a href="https://zijie.wang/" target="_blank">Jay Wang</a>, <a href="https://seongmin.xyz/" target="_blank">Seongmin Lee</a>, <a href="https://bhoov.com/" target="_blank">Benjamin Hoover</a>, and <a href="https://poloclub.github.io/polochau/" target="_blank">Polo Chau</a> at the Georgia Institute of Technology.

## 项目创建人

```bibTeX
@article{cho2024transformer,
  title = {Transformer 解释器: Interactive Learning of Text-Generative Models},
  shorttitle = {Transformer Explainer},
  author = {Cho, Aeree and Kim, Grace C. and Karpekov, Alexander and Helbling, Alec and Wang, Zijie J. and Lee, Seongmin and Hoover, Benjamin and Chau, Duen Horng},
  Chinese ={Lewis Zhang},
  journal={IEEE VIS},
  year={2024}
}


```

## License

The software is available under the [MIT License](https://github.com/poloclub/transformer-explainer/blob/main/LICENSE).

