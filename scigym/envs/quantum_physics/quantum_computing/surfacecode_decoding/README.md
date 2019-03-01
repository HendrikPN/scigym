### _surfacecode_

A surface code environment aimed at facilitating the development of decoding agents for fault tolerant quantum computing.

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/51264006-bfd9ae80-19b5-11e9-8d4c-fb5b757ade76.png" width="600">
</p>  

The environment provided here was previously used for the development of <a href="https://github.com/R-Sweke/DeepQ-Decoding"> deepQ decoders</a> . To get started conceptually, and for an introduction to fault-tolerant quantum computing, and particularly the surface code, we highly suggest reading the associated manuscript <a href="https://arxiv.org/pdf/1810.07207.pdf">Reinforcement Learning Decoders for Fault-Tolerant Quantum Computation</a>.

Additionally, if you use this environment, or any of the code provided here, please cite the above mentioned work as follows:


    R. Sweke, M.S. Kesselring, E.P.L. van Nieuwenburg, J. Eisert,
    Reinforcement Learning Decoders for Fault-Tolerant Quantum Computation,
    arXiv:1810.07207 [quant-ph], 2018. 

Finally, if you use this environment to develop a decoder which outperforms the deepQ decoders described in the above work then please let us know! And have fun :) 

<hr>

#### 1) Documentation

The environment provided here inherits directly from the openAI gym base class gym.env - as such, we recommend starting with the <a href="https://gym.openai.com/docs/"> openAI gym documentation </a> to familiarize yourself with the openAI gym API format.

As the environment has various free attributes which should be specified, an instance of the environment is instantiated as follows:

```python
from scigym.envs.quantum_physics.quantum_computing.surfacecode_decoding import SurfaceCodeEnv
my_env = SurfaceCodeEnv(p_phys=0.001, p_meas=0.001, error_model="X", use_Y=False, volume_depth=1)
```

The attributes which need to be specified are as follows:

1. p_phys and p_meas: The physical and measurement error probabilities respectively. 
2. error_model: A string in ["DP","X"] specifying whether a depolarizing or bit flip channel should be simulated.
3. use_Y: A boolean indicating whether Pauli Y flips are valid actions.
4. volume_depth: A positive integer specifying the number of syndrome measurements performed sequentially in each syndrome extraction.

For more details, we again highly suggest starting by reading <a href="https://arxiv.org/pdf/1810.07207.pdf">Reinforcement Learning Decoders for Fault-Tolerant Quantum Computation</a>. In particular, please note that at the moment the distance of the surface code is fixed to distance-5 - we are working at removing this restriction!

The actions which are allowed depend on the setting of the attributes:

1. If the error model is "X" then any integer in the set [0,25] is a valid action. The integers [0,24] each represent a bit flip on a particular data qubit (labelled sequentially row wise), while the action 25 is the _request new syndrome_ action. In this case num_action_layers = 1.
2. If the error model is "DP" and use_Y is True, then any integer in the set [0,75] is a valid action, with actions in [0,24], [25,49], [50,74] indicating X,Y, or Z flips on the specified qubit respectively, with 75 being the _request new syndrome_ action. In this case num_action_layers = 3.
3. If the error model is "DP" and use_Y is False, then any integer in the set [0,50] is a valid action, with actions in [0,24], [25,49] indicating X or Z flips on the specified qubit respectively, with 50 being the _request new syndrome_ action. In this case num_action_layers = 2.

At any given instant the state of the environment is a [volume_depth + num_action_layers,11,11] boolean array, where:

1. The first volume_depth slices of the tensor encode the most recently obtained syndromes.
2. The final num_action_layers slices of the tensor encode the history of actions taken since the most recent syndrome was obtained.

As an example, for depolarizing noise (i.e. error_model = "DP"), with use_Y=False and volume_depth = 3, the state of the environment would be as follows:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/51263374-6f158600-19b4-11e9-8f87-517200f0698f.png" width="700">
</p>  

with the syndrome layers utilizing the following encoding:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/51263373-6f158600-19b4-11e9-84d6-d83196b900fa.png" width="500">
</p>  

and the action-history layers utilizing this encoding.

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/51263376-6f158600-19b4-11e9-9e81-206f7f25f13d.png" width="500">
</p>  


#### 2) To-Do and Future Development

At the moment the most obvious restriction of this environment is that it is restricted to distance-5 surface codes. This restriction is purely because of the feed-forward homology class predicting referee decoders that have been used. This restriction could be lifted by using minimum-weight perfect matching decoders as referees, which is in the process of being implemented.

If you have good results for environment provided here, and we haven't yet removed the restriction to d=5, please contact us! We are very interested in developing scalable decoders!!
