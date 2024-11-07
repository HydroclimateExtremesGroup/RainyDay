# RainyDay
RainyDay Rainfall Hazard Analysis System

NOTE: We are in the process of a major refactoring of RainyDay. If you wish to use the code and compatible data, please contact Daniel Wright (danielb.wright@wisc.edu).

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.  I will attempt to keep the documentation in synch with major changes to the code.  I would appreciate any feedback on the user guide and on RainyDay itself, so I can continue to improve both.

Please note also that this repository does not contain any of the NetCDF-formatted precipitation files that are needed to actually perform an analysis with RainyDay. If you are interested in performing an analysis, I would recommend contacting me so we can discuss which datasets I have available in the proper input format. A web-based demo version, with extremely limited options, is available here: https://her.cee.wisc.edu/rainyday-rainfall-for-modern-flood-hazard-assessment/. Development of the web-based version has been supported by the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT

