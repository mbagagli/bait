# **Ba.** _It_
**It**erative **Ba**er picking system.

*  AUTHOR: _Matteo Bagagli @ ETH-Zurich_
* VERSION: _2.5.4_
*    DATE: _09/2019_

----------

The BaIt picking system is a software created around the already famous and widely used Baer-Kradolfer picker (Baer 1987). The proposed seismic picker push forward the already great performance of the Baer-Kradolfer algorithm by adding an iterative picking procedure over the given seismic trace.

It rely on standard libraries like numpy, matplotlib and ObsPy. For the installation, see further.

## Installation
The file `./data/bait_condaenv.ylm` contains all the encessary dependencies to install and run the framework. The lines to be typed in the terminal are

```bash
$ conda create -n bait python=3.6
$ conda activate bait
$ pip install .
$ pytest # to check everything is cool :)
```

Prior of the run of the subsequent code, the user must have installed `conda` or `miniconda` ([HOW-TO](https://conda.io/docs/user-guide/install/index.html)).

----------
#### References

- Baer, M., and U. Kradolfer. "An automatic phase picker for local and teleseismic events." Bulletin of the Seismological Society of America 77.4 (1987): 1437-1445.

##### Funny Quotes
- Always code as if the guy who ends up maintaining your code will be a violent psycopath who knows where you live (Jhon Woods)
