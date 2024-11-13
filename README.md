# RainyDay
RainyDay Rainfall Hazard Analysis System

NOTE: We are in the process of a major refactoring of RainyDay. If you wish to use the code and compatible data, please contact Daniel Wright (danielb.wright@wisc.edu).

Please follow these instructions for accessing  AORC data:
How to download AORC data from HER S3
 
1. Open your terminal (Mac or Linux) or terminal emulator such as Cygwin (on Windows; must have ‘wget’ installed).

2. Download the file containing the URLs of all AORC files using the following command. This text file lists all the URLs for AORC files from February 1, 1979, to December 31, 2021: wget https://web.s3.wisc.edu/aorc/aorc_urls.txt

3. Filter the URLs to include only those for the files you wish to download, and then save them in a new file, such as download.txt. Below are some examples:

     -Subset the files for 1980: grep "1980/" aorc_urls.txt > download.txt

     -Subset the files for January 1980: grep "1980/AORC.198001" aorc_urls.txt > download.txt

     -Subset the files from 1980 to 1985: grep -E "1980/AORC.1980|1981/AORC.1981|1982/AORC.1982|1983/AORC.1983|1984/AORC.1984|1985/AORC.1985" aorc_urls.txt > download.txt

4. Use the wget command with download.txt to download the files: wget -i 1980_urls.txt

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.  I will attempt to keep the documentation in synch with major changes to the code.  I would appreciate any feedback on the user guide and on RainyDay itself, so I can continue to improve both.

Development of the web-based version has been supported by the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT

