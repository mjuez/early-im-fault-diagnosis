# Early and extremely early multi-label fault diagnosis in induction motors

https://doi.org/10.1016/j.isatra.2020.07.002

Mario Juez-Gil <sup>a</sup>, Juan José Saucedo-Dorantes <sup>b</sup>, Álvar Arnaiz-González <sup>a</sup>, César Ignacio García-Osorio <sup>a</sup>, Carlos López-Nozal <sup>a</sup>, David Lowe <sup>c</sup>.

<sup>a</sup> Universidad de Burgos, Burgos, Spain

<sup>b</sup> Autonomous University of Queretaro, Mexico

<sup>c</sup> Aston University, Birmingham, United Kingdom

## Abstract

The detection of faulty machinery and its automated diagnosis is an industrial priority because efficient fault diagnosis implies efficient management of the maintenance times, reduction of energy consumption, reduction in overall costs and, most importantly, the availability of the machinery is ensured. Thus, this paper presents a new intelligent multi-fault diagnosis method based on multiple sensor information for assessing the occurrence of single, combined, and simultaneous faulty conditions in an induction motor. The contribution and novelty of the proposed method include the consideration of different physical magnitudes such as vibrations, stator currents, voltages, and rotational speed as a meaningful source of information of the machine condition. Moreover, for each available physical magnitude, the reduction of the original number of attributes through the Principal Component Analysis leads to retain a reduced number of significant features that allows achieving the final diagnosis outcome by a multi-label classification tree. The effectiveness of the method was validated by using a complete set of experimental data acquired from a laboratory electromechanical system, where a healthy and seven faulty scenarios were assessed. Also, the interpretation of the results do not require any prior expert knowledge and the robustness of this proposal allows its application in industrial applications, since it may deal with different operating conditions such as different loads and operating frequencies. Finally, the performance was evaluated using multi-label measures, which to the best of our knowledge, is an innovative development in the field condition monitoring and fault identification.

## Experiments

The experiments are available in [this notebook](experiments.ipynb).

## Aknowlegments

This work was supported through project TIN2015-67534-P (MINECO, Spain/FEDER, UE) of the Ministerio de Economía y Competitividad of the Spanish Government, project BU085P17 (JCyL/FEDER, UE) of the Consejería de Educación of the Junta de Castilla y León, Spain (both projects co-financed through European Union FEDER funds), and by the pre-doctoral grant (EDU/1100/2017), also of the Consejería de Educación of the Junta de Castilla y León, Spain and the European Social Fund. The authors gratefully acknowledge the support of NVIDIA Corporation, United States and its donation of the TITAN Xp GPUs used in this research.



## Citation policy

Please cite this research as:

```
@ARTICLE{juezgil2020earlyimfaults,
title = {Early and extremely early multi-label fault diagnosis in induction motors},
author = {Juez-Gil, Mario and Saucedo-Dorantes, Juan Jos{\'e} and Arnaiz-Gonz{\'a}lez, {\'A}lvar and L{\'o}pez-Nozal, Carlos and Garc{\'\i}a-Osorio, C{\'e}sar and David Lowe},
journal = {ISA Transactions},
year = {2020},
month = {nov},
volume = {106},
pages = {367-381},
issn = {0019-0578},
doi = {https://doi.org/10.1016/j.isatra.2020.07.002},
url = {http://www.sciencedirect.com/science/article/pii/S0019057820302755},
keywords = {Multi-fault detection, Early detection, Multi-label classification, Principal component analysis, Load insensitive model, Prediction at low operating frequencies},
publisher = {Elsevier}
}
```
