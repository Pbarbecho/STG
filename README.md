<p align="left">
  <img src="doc/logo.png" width="400">
</p>

# Automating OMNeT++ simulations #

OSM allows to OMNeT++ users to quickly and easily execute large-scale network simulations. 
This is an automation tool for OMNeT++ large-scale simulations and data analysis.
Based on OMNeT++ structure, this tool reads .ini file and build simulation campaign.
Users' manual an code documentation is available at [readthedocs][rtd].
 
How to cite us 
--------------

If you use SMO for your OMNeT++ experiment analysis, we would appreciate a citation of our work:

* P. Bautista, L. F. Urquiza-Aguiar, L. L. Cárdenas and M. A. Igartua, “Large-Scale Simulations Manager Tool for OMNeT++: Expediting Simulations and Post-Processing Analysis,” in IEEE Access, vol. 8, pp. 159291-159306, 2020, doi: 10.1109/ACCESS.2020.3020745.


Feature highlights 
------------------

* Supports Python >= 3.5;
* Fine grane control of the simulation campaign;
* Customizable/interactive plotting
* Runs parallelized simulations and post-processing for large number of files (common in large-scale simulations);



OSM includes the following tool:    
```bash
  - Launcher: build simulation campaign and execute parallel simulations in batches.
  - Parser:   Automatically try to detect output results files from simulation campaign (.vec,.sca, custom format) and convert those to an unique output file. 
  - Analyzer: Reads parsed files and plot results from template or launch an interactive plot in a web browser (pyvot tables). 
```

```bash
  # Build and lauch the simulation campaign
  $osm launcher [OPTIONS] INIFILE MAKEFILE

  # Summarize result files located in output folder
  $osm summarizer [OPTIONS] 

  # Analyze summarized file 
  $osm analyzer [OPTIONS] 
```

[![Documentation Status](https://readthedocs.org/projects/osm/badge/?version=latest)](https://osm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Pbarbecho/osm/branch/master/graph/badge.svg)](https://codecov.io/gh/Pbarbecho/osm)


## Clone the repository ##
The osm package is developed using a pipenv. TO install osm on a virtual environment:
```bash
pip3 install pipenv
```

To clone the osm repository, on the command line, enter:
```bash
git clone https://github.com/Pbarbecho/osm.git
```
On the new venv, install osm project in an editable mode:

```bash
pipenv install -e osm/
```

Then, for use the new virtual environment instantiate a sub-shell as follows:

```bash
pipenv shell
```

At this time, you can interact with the osm modules, customize you analysis and use osm utilities. 

## Downloading modules ##

Install osm using pip:
```bash
pip3 install --user -U https://github.com/Pbarbecho/osm/archive/master.zip
```

Depending on the operating system, you may need to add ~/.local/bin to your path. During the installation you will be warned about this.
 
 
In case you want to uninstall osm package: 

```bash
pip3 uninstall osm
```

## Authors ##

Pablo Barbecho

[rtd]: https://osm.readthedocs.io/en/latest/
