# MADS_Milestone2_Bird_Migration


Checklist data can be found at:
https://drive.google.com/file/d/1t-F-3YYgX5y-vuxrvMY8CCfe4OWP7iuZ/view?usp=share_link

Species occurrence data can be found at:
https://drive.google.com/drive/folders/1eITjXzp71Zlcs89hbfJ1UatHRqnjESL3?usp=share_link

Final report for the project can be found at:
https://docs.google.com/document/d/1et1vahsuMA7iXkstRpjRU9MsPGmSWWPP/edit?usp=share_link&ouid=106891747267994283867&rtpof=true&sd=true



# SIADS 696 Milestone II - Final Report
Project Name: Predicting and Clustering Bird Migration Patterns Across Americas
Team: Yangkang Chen, Dean Lawrence 

## Introduction

Background
In this project, our goal is to predict, cluster, and compare different bird migration patterns.

Throughout the years, millions of birds were observed and recorded across the world by birders at eBird citizen science project. However, we still know little regarding the detailed migration pattern of birds. On the one hand, the global climate and land-use change is threatening birds, and on the other hand the spatial-temporal nature of bird migration makes it challenging to predict movement. Therefore, there is an urgency to model fine-scale bird migration and distribution patterns across the continents, both for understanding and conservation purposes. In this project, we focused on fitting the current migration pattern, so that we could predict birds' presence or not in regions where there is no observation data available. 

Goal and purpose
Through this project, we could:
1) Predict the presence or absence of a species at a location at a time, even in a region where there is no observation data available. This will provide a comprehensive graph of the whole-year-round bird distribution pattern.
2) By taking spatial-temporal connectivity into account, we could depict the migration route of certain species, which is significant in understanding the biological process and conservation.
3) The fitted model could further be used to project potential occurrence change under future land-use and climate scenarios to better quantify the impact of global change on migratory birds.

Methods & results summary
For supervised learning, we compared the metrics for 56different model selection, including five baseline models and one advanced ensemble model. We showed that the gridding-ensemble method (here after, AdaSTEM model) is generally much better than single baseline models. Prediction difficulty for different species is different, with House Wren have the highest AUC score of 0.8646 in AdaSTEM model, and Mallard with the lowest of 0.8615.

In unsupervised learning part, we first calculated the geographical center of distribution for each calendar week/month, and generated a "migration route" for the species. Then we applied different clustering strategies to depict similarity and dissimilarity of species migration pattern. We found that species in closer evolutionary relationship or with similar tropical niche migrate together. For example, cluster 1 mainly consists of carnivorous predator, while cluster 2 mainly consists of omnivorous and vegetarian preys.


## Data Source
  eBird citizen science data (https://science.ebird.org/en/use-ebird-data)
	    Time range: year 2018.
	    Originally 4,300,429 observations. 487,293 after subsampled.
	    Important variables:
	        Time of the day when observation started.
	        Date
	        Number of observers
          Observation protocol type (stationary or traveling)
          Location (location name and longitude, latitude)
          Traveling distance
          The name of each species observed and their count.
          
      eBird data were pre-filtered based on following rules:
          Observation type should be traveling or stationary.
          Only checklists with more than 5 species observed are included.
          The travel distance of observer should be less than 3km (to make sure the high spatial precision and land-use continuity)
          The observation time duration should be more than 5 minutes.
          
  ESA CCI global land use data (https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover?tab=form)
      Time range: year 2018.
	    Important variables include: Fraction of land use, landscape index (maximum patch size, patch density, etc.) of different land use (urban, cropland, shrubland, forest, water, etc.)
	
  WorldClim monthly climate data (https://www.worldclim.org)
	    Time range: year 2018.
	    19 Bioclimatic variables. For example, precipitation of the wettest quarter, temperature of the warmest month.
	    Monthly raw climate data: maximum temperature (tmax), minimum temperature (tmin), precipitation (prec).
	
  Elevation, slope data (https://www.earthenv.org/topography)
	    Mean and standard deviation of elevation, slope, eastness, northness.


































