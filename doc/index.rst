.. ovsm documentation master file, created by
   sphinx-quickstart on Tue Mar 17 11:23:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Automating large-scale simulations for OMNeT++
==============================================

OSM allows to OMNeT++ users to quickly and easily execute large-scale network simulations. Three shell commands (including help context) are available:


.. code:: bash

  # Build and lauch the simulation campaign
  $osm launcher [OPTIONS] INIFILE MAKEFILE

  # Summarize result files located in output folder
  $osm parser [OPTIONS]

  # Analyze summarized file
  $osm analyzer [OPTIONS]

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


User's guide
------------
.. toctree::
   :maxdepth: 2

   installation
   cli

Code Documentation
--------------------
.. toctree::
   :maxdepth: 2

   modules

Links
------
   Github: `<https://github.com/Pbarbecho/osm>`_
