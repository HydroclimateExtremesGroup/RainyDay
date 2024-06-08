# RainyDay
RainyDay Rainfall Hazard Analysis System

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.  I will attempt to keep the documentation in synch with major changes to the code.  I would appreciate any feedback on the user guide and on RainyDay itself, so I can continue to improve both.

Please note also that this repository does not contain any of the NetCDF-formatted precipitation files that are needed to actually perform an analysis with RainyDay. If you are interested in performing an analysis, I would recommend contacting me so we can discuss which datasets I have available in the proper input format. A web-based demo version, with extremely limited options, is available here: https://her.cee.wisc.edu/rainyday-rainfall-for-modern-flood-hazard-assessment/. Development of the web-based version has been supported by the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT

Several precipitation datasets have been prepared for usage in RainyDay. Each has its pros and cons in terms of accuracy, record length, and resolution. You should carefully consider the needs of your particular application. These datasets are provided without any guarantee of accuracy. In the case of NCEP Stage IV, it has been regridded from the HRAP to a geographic projection, and thus the actual precipitation rates vary slightly from the original dataset. 

If you want RainyDay-ready Stage IV precipitation data for CONUS, see here: https://drive.google.com/drive/folders/1g6_Cvxnr4IGqqzDrKWv9YquUG5_pZINK?usp=sharing

If you want RainyDay-ready NLDAS-2 precipitation data for CONUS, see here: https://drive.google.com/drive/folders/1if8-rZk-qvqztjhYtscxpl_nmBDW64YQ?usp=sharing

If you want RainyDay-ready nClimGrid-Daily precipitation data for CONUS, see here:https://drive.google.com/drive/folders/1G9-qc3gQgwuvmK_gAlNdgPOR111R345z?usp=sharing

If you want RainyDay-ready IMERG V06B Final run data for the entire globe, see here (warning, this dataset is large; about 320 GB): https://drive.google.com/drive/folders/1b1I9BU-48lF-3Gavvt5JIwXy0wtC0rsU?usp=sharing
