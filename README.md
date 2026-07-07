# AuGR: Augmenting Generative-Ranking Joint Training for Chatbot Intent Recommendation

This is the official repository for the AuGR paper. This repository contains code to replicate the results obtained within the paper for public benchmarks, for both sequential recommendation and CTR prediction tasks.

## Results on Public Benchmarks

### Sequential Recommendation (Amazon 2014)
Only AuGR variants were ran by us, results from other models were obtained from [https://phonism.github.io/genrec](https://phonism.github.io/genrec).

#### Beauty:
| Model | Backbone | R@5 | R@10 | N@5 | N@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SASRec(CE) | - | 0.0538 | 0.0851 | 0.0320 | 0.0421 |
| HSTU (CE) | - | 0.0568 | 0.0859 | 0.0347 | 0.0441 |
| TIGER | - | 0.0419 | 0.0644 | 0.0282 | 0.0354 |
| LCRec | - | 0.0481 | 0.0704 | 0.0331 | 0.0403 |
| OneRec-SFT | - | **0.0578** | 0.0816 | **0.0398** | **0.0475** |
| AuGR | TIGER | 0.0402 | 0.0639 | 0.0260 | 0.0336 |
| AuGR | HSTU | 0.0559 | **0.0889** | 0.0355 | 0.0462 |

#### Toys:
| Model | Backbone | R@5 | R@10 | N@5 | N@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SASRec(CE) | - | 0.0613 | 0.0922 | 0.0348 | 0.0448 |
| HSTU (CE) | - | 0.0611 | 0.0914 | 0.0363 | 0.0461 |
| TIGER | - | 0.0340 | 0.0521 | 0.0214 | 0.0272 |
| LCRec | - | 0.0433 | 0.0614 | 0.0310 | 0.0368 |
| OneRec-SFT | - | 0.0545 | 0.0790 | 0.0383 | 0.0462 |
| AuGR | TIGER | 0.0354 | 0.0557 | 0.0225 | 0.0291 |
| AuGR | HSTU | **0.0631** | **0.0934** | **0.0381** | **0.0479** |

#### Sports:
| Model | Backbone | R@5 | R@10 | N@5 | N@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SASRec(CE) | - | 0.0321 | 0.0495 | 0.0191 | 0.0248 |
| HSTU (CE) | - | 0.0283 | 0.0439 | 0.0182 | 0.0232 |
| TIGER | - | 0.0236 | 0.0377 | 0.0150 | 0.0195 |
| LCRec | - | 0.0238 | 0.0360 | 0.0159 | 0.0198 |
| OneRec-SFT | - | 0.0299 | 0.0436 | 0.0200 | 0.0244 |
| AuGR | TIGER | 0.0251 | 0.0399 | 0.0154 | 0.0201 |
| AuGR | HSTU | **0.0326** | **0.0509** | **0.0205** | **0.0265** |

### CTR Prediction (TaoBao Ads)
| Model | AUC | LogLoss |
| :--- | :--- | :--- |
| SASRec | 0.5964 | 0.2028 |
| AuGR-SASRec | **0.5996** | 0.2060 |
| HSTU | 0.5978 | 0.2013 |
| AuGR-HSTU | **0.5994** | 0.2044 |

## Reproducing the results
Refer to the respective markdowns for the setup instructions.
- CTR Prediction: [CTR Prediction README](https://github.com/RoydonTay/AuGR/tree/main/ctr-prediction)
- Sequential Recommendation: [Sequential Recommendation README](https://github.com/RoydonTay/AuGR/tree/main/generative-recommendation)

## References
- [https://phonism.github.io/genrec](https://phonism.github.io/genrec): GenRec: A Model Zoo for Generative Recommendation

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@misc{augrexperimentcode,
  title = {AuGR: Augmenting Generative-Ranking Joint Training for Chatbot Intent Recommendation},
  author = {Tingting Yu, Roydon Tay, Cheng Chang, Koh Zhi Rong, Bin Fu, Kwan Hui Lim},
  year = {2026},
  eprint={...},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url = {...}
}
```
