# DSTC10 Automatic Evaluation of Open-domain Dialogue Systems

In this task, our goal is to seek effective automatic dialogue evaluation metrics that correlates well with human judgements and that are explainable. These metrics can serve as a proxy to human evaluation for fast prototyping of open-domain chatbots.

## Dataset
Please register and download the data at [here](https://chateval.org/dstc10). Once downloaded, unzip the human_evaluation_data.zip at the current folder. The dataset is a validation set to test the effectiveness of the proposed metrics. It consists of the following 14 components:

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

| Dataset                   | No. Turns/Dialogues  | No. Anno Qualities  | No. Annos  | AVG. Utts  | AVG. Words per Utts  |           
|-------------------------- |--------------------- |---------------------|------------|------------|----------------------|
| DSTC6-Eval (D6)           | 40000                |    1                | 400000     | 2.63       | 11.36                |
| DSTC7-Eval (D7)           | 9900                 |    1                | 29700      | 4.92       | 20.18                |
| Persona-Chatlog (PC)      | 3316                 |    9                | 29844      | 12.00      | 7.59                 |
| PersonaChat-USR (UP)      | 300                  |    6                | 5400       | 9.30       | 11.87                |
| TopicalChat-USR (TP)      | 360                  |    6                | 6480       | 11.20      | 23.14                |
| FED-Turn (FT)             | 375                  |    9                | 3348       | 10.37      | 9.70                 |
| FED-Conversation (FC)     | 125                  |    11               | 1364       | 12.72      | 8.70                 |
| DailyDialog-Gupta (GD)    | 500                  |    1                | 1500       | 4.92       | 12.36                |
| DailyDialog-Zhao (ZD)     | 900                  |    4                | 14400      | 4.72       | 12.39                |
| PersonaChat-Zhao (ZP)     | 900                  |    1                | 3600       | 5.13       | 11.77                |
| DailyDialog-Grade (ED)    | 300                  |    1                | 3000       | 3.00       | 12.25                |
| Empathetic-Grade (EE)     | 300                  |    1                | 3000       | 3.00       | 14.86                |
| ConvAI2-Grade (EC)        | 300                  |    1                | 3000       | 3.00       | 11.89                |
| HUMOD (HU)                | 9500                 |    2                | 57000      | 3.95       | 4.31                 |

### Data Meta-information

| Dataset                   | Contains References ? | Multiple References ? | Annotation Granularity    |         
|-------------------------- |-----------------------|-----------------------|---------------------------|
| DSTC6-Eval (D6)           | Yes                   | Yes                   | Turn-level                |
| DSTC7-Eval (D7)           | Yes                   | No                    | Turn-level                |
| Persona-Chatlog (PC)      | No                    | -                     | Dialogue-level            |
| PersonaChat-USR (UP)      | Yes                   | No                    | Turn-level                |
| TopicalChat-USR (TP)      | Yes                   | No                    | Turn-level                |
| FED-Turn (FT)             | No                    | -                     | Turn-level                |
| FED-Conversation (FC)     | No                    | -                     | Dialogue-level            |
| DailyDialog-Gupta (GD)    | Yes                   | Yes                   | Turn-level                |
| DailyDialog-Zhao (ZD)     | Yes                   | No                    | Turn-level                |
| PersonaChat-Zhao (ZP)     | Yes                   | No                    | Turn-level                |
| DailyDialog-Grade (ED)    | No                    | -                     | Turn-level                |
| Empathetic-Grade (EE)     | No                    | -                     | Turn-level                |
| ConvAI2-Grade (EC)        | No                    | -                     | Turn-level                |
| HUMOD (HU)                | Yes                   | Yes                   | Turn-level                |

### JSON Data Formats

#### Turn-level

The *xxx_eval.json* file includes the list of instances each of which is a context-response pair data point.
Key components of each instance :

* dialogue_id: the unique id assigned to each data instance
* model: name of system that generated the response based on the context
* context: the dialogue context delimited by *\n* token
* response: the corresponding system response following the context
* reference: list of human-written reference responses w.r.t the context
* annotations: 
  {
    * [dialogue quality]: list of scores provided by annotators
  }
 
#### Dialogue-level

The *xxx_eval.json* file includes the list of instances each of which is a single conversation.
Key components of each instance :

* dialogue_id: the unique id assigned to each data instance
* model: name of system that generated the response based on the context
* dialogue (list of utterances): 
  [
    * {speaker: xxx, text: xxx}
  ]
* annotations: 
  {
    * [dialogue quality]: list of scores provided by annotators
  } 

Note that for UP, TP and PC, there are additional information w.r.t facts or personas associated with the dialogues. Participants may consider using these information to design their metrics.

## How will we rank all the submitted metrics in the leaderboard?

During development phase

* We will first average the Spearman correlation scores of the submitted metric within the dataset.
* Next, all the dataset-wise average Spearman correlation scores will be averaged across all the 14 datasets.
* The submitted metrics will be ranked based on the final single Spearman correlation score.

During the final evaluation phase

* We will adopt a weighted average approach to determine the final ranking of the submitted metrics based on their performance on the validation set as well as the hidden test set which will be released after the development phase. A high weightage will be given to the metrics' performance on the hidden test set.

Note that it is not necessary to have a single metric score for all the annotated dialogue qualities. Besides high correlation with human judgements, we also encourage explainability of the metrics.

## Timeline
* Validation data released: Jun 14, 2021
* Test data released: Sep 13, 2021
* Entry submission deadline: Sep 21, 2021
* Final result announcement: Oct 1, 2021 - Oct 8, 2021

## Test Data & Results

* Both test data and results can be found at [here](https://chateval.org/dstc10)
* Detailed results of all submissions can be found at [here](https://docs.google.com/spreadsheets/d/10yl4-tDFEroa_qZsC4Fv_NO2TWjHO31Xv847rWVhPQ8/edit#gid=378909727)

## Organizers
- Chen Zhang (National University of Singapore, Singapore)
- Haizhou Li (National University of Singapore, Singapore)
- Jo??o Sedoc (New York University, USA)
- Luis F. D'Haro (Universidad Polit??cnica de Madrid, Spain)
- Rafael Banchs (Intapp Inc., USA)
- Alexander Rudnicky (Carnegie Mellon University, USA)

## References
  <p>[1] Deriu, J., Rodrigo, A., Otegi, A., Echegoyen, G., Rosset, S., Agirre, E., & Cieliebak, M. (2020). Survey on evaluation methods for dialogue systems. Artificial Intelligence Review, 1-56.</p>
  <p>[2] Hori, C., & Hori, T. (2017). End-to-end conversation modeling track in DSTC6. arXiv preprint arXiv:1706.07440.</p>
  <p>[3] Galley, M., Brockett, C., Gao, X., Gao, J., & Dolan, B. (2019). Grounded response generation task at dstc7. In AAAI Dialog System Technology Challenges Workshop.</p>
  <p>[4] See, A., Roller, S., Kiela, D., & Weston, J. (2019, June). What makes a good conversation? How controllable attributes affect human judgments. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 1702-1723).</p>
  <p>[5] Mehri, S., & Eskenazi, M. (2020). USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation. arXiv preprint arXiv:2005.00456.</p>
  <p>[6] Mehri, S., & Eskenazi, M. (2020, July). Unsupervised Evaluation of Interactive Dialog with DialoGPT. In Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue (pp. 225-235).</p>
  <p>[7] Zhang C., D???Haro L.F., Banchs R.E., Friedrichs T., Li H. (2021) Deep AM-FM: Toolkit for Automatic Dialogue Evaluation. In Conversational Dialogue Systems for the Next Decade. Lecture Notes in Electrical Engineering, vol 704. Springer, Singapore.</p>
  <p>[8] Zhao, T., Lala, D., & Kawahara, T. (2020, July). Designing Precise and Robust Dialogue Response Evaluators. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 26-33).</p>
  <p>[9] Gupta, P., Mehri, S., Zhao, T., Pavel, A., Eskenazi, M., & Bigham, J. P. (2019, September). Investigating Evaluation of Open-Domain Dialogue Systems With Human Generated Multiple References. In Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue (pp. 379-391).</p>
  <p>[10] Huang, L., Ye, Z., Qin, J., Lin, L., & Liang, X. (2020, November). GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 9230-9240).</p>
  <p>[11] Merdivan, E., Singh, D., Hanke, S., Kropf, J., Holzinger, A., & Geist, M. (2020). Human annotated dialogues dataset for natural conversational agents. Applied Sciences, 10(3), 762.</p>
  <p>&nbsp;</p>
