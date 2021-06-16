# DSTC10 Automatic Evaluation of Open-domain Dialogue Systems

In this task, our goal is to seek effective automatic dialogue evaluation metrics that correlates well with human judgements and that are explainable. These metrics can serve as a proxy to human evaluation for fast prototyping of open-domain chatbots.

## Dataset
Please register and download the data at https://chateval.org/dstc10. Once downloaded, unzip the human_evaluation_data.zip at the current folder. The dataset is a validation set to test the effectiveness of the proposed metrics. It consists of the following 14 components:

1. DSTC6-Eval (D6) (Hori et al., 2017)
2. DSTC7-Eval (D7) (Galley et al., 2019)
3. Persona-Chatlog (PC) (See et al., 2019)
4. PersonaChat-USR (UP) (Mehri & Eskenazi, 2020a)
5. TopicalChat-USR (TP) (Mehri & Eskenazi, 2020a)
6. FED-Turn (FT) (Mehri & Eskenazi, 2020b)
7. FED-Conversation (FC) (Mehri & Eskenazi, 2020b)
8. DailyDialog-Eval (GD) (Gupta et al., 2019)
9. DailyDialog-Eval (ZD) (Zhao et al., 2020)
10. PersonaChat-Eval (ZP) (Zhao et al., 2020)
11. DailyDialog-Eval (ED) (Huang et al., 2020)
12. Empathetic-Eval (EE) (Huang et al., 2020)
13. ConvAI2-Eval (EC)  (Huang et al., 2020)
14. HUMOD (HU) (Merdivan et al., 2020)

### Data Statistics

| Dataset                   | No. Turns/Dialogues  | No. Anno Qualites  | No. Annos  | AVG. Utts  | AVG. Words per Utts  |          
|-------------------------- |-------------------   |--------------------|------------|------------|----------------------|
| DSTC6-Eval (D6)           | 40000                |    1               | 400000     | 2.63       | 12.36                |
| DSTC7-Eval (D7)           | 500                  |    1               | 1500       | 4.92       | 20.18                |
| Persona-Chatlog (PC)\*\*  | 3316                 |    9               | 29844      | 12.00      | 7.59                 |
| PersonaChat-USR (UP)      | 300                  |    6               | 5400       | 9.30       | 11.87                |
| TopicalChat-USR (TP)      | 360                  |    6               | 6480       | 11.20      | 23.14                |
| FED-Turn (FT)*            | 375                  |    9               | 3348       | 10.37      | 9.70                 |
| FED-Conversation (FC)\*\* | 125                  |    11              | 1364       | 12.72      | 8.70                 |
| DailyDialog-Eval (GD)     | 500                  |    1               | 1500       | 4.92       | 12.36                |
| DailyDialog-Eval (ZD)     | 900                  |    4               | 14400      | 4.72       | 13.39                |
| PersonaChat-Eval (ZP)     | 900                  |    1               | 3600       | 5.13       | 12.77                |
| DailyDialog-Eval (ED)\*   | 300                  |    1               | 3000       | 3.00       | 12.25                |
| Empathetic-Eval (EE)\*    | 300                  |    1               | 3000       | 3.00       | 14.86                |
| ConvAI2-Eval (EC)*        | 300                  |    1               | 3000       | 3.00       | 11.89                |
| HUMOD (HU)                | 9500                 |    2               | 57000      | 2.00       | 14.51                |

\* indicates no reference response <br />
\*\* denotes interactive dialogue-level evalaution dataset 

### JSON Data Formats

#### Turn-level Evaluation

The *xxx_eval.json* file includes the list of instances each of which is a context-response pair data point.
Key components of each instance :

* context: the dialogue context delimited by *\n* token
* response: the corresponding system response following the context
* annotations: 
  {
    * [annotation quality]: list of scores
 * }
 
#### Dialogue-level Evaluation

The *xxx_eval.json* file includes the list of instances each of which is a single conversation.
Key components of each instance :

* dialogue (list of utterances): 
  [
    * {speaker: xxx, text: xxx}
  * ]
* annotations: 
  {
    * [annotation quality]: list of scores
 * } 


## Organizers
- Chen Zhang (National University of Singapore, Singapore)
- Haizhou Li (National University of Singapore, Singapore)
- João Sedoc (New York University, USA)
- Luis F. D'Haro (Universidad Politécnica de Madrid, Spain)
- Rafael Banchs (Intapp Inc., USA)
- Alexander Rudnicky (Carnegie Mellon University, USA)

## References
  <p>[1] Deriu, J., Rodrigo, A., Otegi, A., Echegoyen, G., Rosset, S., Agirre, E., & Cieliebak, M. (2020). Survey on evaluation methods for dialogue systems. Artificial Intelligence Review, 1-56.</p>
  <p>[2] Hori, C., & Hori, T. (2017). End-to-end conversation modeling track in DSTC6. arXiv preprint arXiv:1706.07440.</p>
  <p>[3] Galley, M., Brockett, C., Gao, X., Gao, J., & Dolan, B. (2019). Grounded response generation task at dstc7. In AAAI Dialog System Technology Challenges Workshop.</p>
  <p>[4] See, A., Roller, S., Kiela, D., & Weston, J. (2019, June). What makes a good conversation? How controllable attributes affect human judgments. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 1702-1723).</p>
  <p>[5] Mehri, S., & Eskenazi, M. (2020). USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation. arXiv preprint arXiv:2005.00456.</p>
  <p>[6] Mehri, S., & Eskenazi, M. (2020, July). Unsupervised Evaluation of Interactive Dialog with DialoGPT. In Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue (pp. 225-235).</p>
  <p>[7] Zhang C., D’Haro L.F., Banchs R.E., Friedrichs T., Li H. (2021) Deep AM-FM: Toolkit for Automatic Dialogue Evaluation. In Conversational Dialogue Systems for the Next Decade. Lecture Notes in Electrical Engineering, vol 704. Springer, Singapore.</p>
  <p>[8] Zhao, T., Lala, D., & Kawahara, T. (2020, July). Designing Precise and Robust Dialogue Response Evaluators. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 26-33).</p>
  <p>[9] Gupta, P., Mehri, S., Zhao, T., Pavel, A., Eskenazi, M., & Bigham, J. P. (2019, September). Investigating Evaluation of Open-Domain Dialogue Systems With Human Generated Multiple References. In Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue (pp. 379-391).</p>
  <p>[10] Huang, L., Ye, Z., Qin, J., Lin, L., & Liang, X. (2020, November). GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 9230-9240).</p>
  <p>[11] Merdivan, E., Singh, D., Hanke, S., Kropf, J., Holzinger, A., & Geist, M. (2020). Human annotated dialogues dataset for natural conversational agents. Applied Sciences, 10(3), 762.</p>
  <p>&nbsp;</p>
