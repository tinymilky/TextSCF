# TextSCF: LLM-Enhanced Image Registration Model

![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2111.10480-b31b1b.svg)](https://arxiv.org/abs/2311.15607)

TextSCF is a comprehensive library designed for weakly supervised image alignment and registration, providing extensive utilities for detailed deformation field analysis.

## Updates

🔥[11/30/2023] - We collect a list of papers exploring the use of LLMs in AI for medicine and healthcare. ([Awesome-Medical-LLMs](docs/Awesome-Medical-LLMs.md)) <br>
🔥[11/30/2023] - We collect a list of papers centered on Image Registration Models in Healthcare. ([Awesome-Medical-Image-Registration](docs/Awesome-Medical-Image-Registration.md))

## Papers

**[Spatially Covariant Image Registration with Text Prompts](https://arxiv.org/abs/2311.15607)** <br>
[Hang Zhang](https://tinymilky.com), Xiang Chen, Rongguang Wang, Renjiu Hu, [Dongdong Liu](https://ddliu365.github.io/), and [Gaolei Li](https://icst.sjtu.edu.cn/DirectoryDetail.aspx?id=28). <br>
arXiv 2023. 

**[Spatially Covariant Lesion Segmentation](https://www.ijcai.org/proceedings/2023/0190)**  <br>
[Hang Zhang](https://tinymilky.com), Rongguang Wang, Jinwei Zhang, [Dongdong Liu](https://ddliu365.github.io/), Chao Li, and Jiahao Li.  <br>
IJCAI 2023.

## Todo
- [x] Awesome-Medical-LLMs
- [x] Awesome-Medical-Image-Registration
- [ ] Core code release
- [ ] Pretrained model release
- [ ] Tutorials and periphery code for smoothness and complexity analysis
- [ ] Tutorials and periphery code for statistical analysis
- [ ] Tutorials and periphery code for model expansibility
- [ ] Tutorials and periphery code for discontinuity-preserving deformation field

## Citation
If our work has influenced or contributed to your research, please kindly acknowledge it by citing:
```
@misc{zhang2023spatially,
      title={Spatially Covariant Image Registration with Text Prompts}, 
      author={Hang Zhang and Xiang Chen and Rongguang Wang and Renjiu Hu and Dongdong Liu and Gaolei Li},
      year={2023},
      eprint={2311.15607},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

@inproceedings{ijcai2023p0190,
  title     = {Spatially Covariant Lesion Segmentation},
  author    = {Zhang, Hang and Wang, Rongguang and Zhang, Jinwei and Liu, Dongdong and Li, Chao and Li, Jiahao},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {1713--1721},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/190},
  url       = {https://doi.org/10.24963/ijcai.2023/190},
}

```