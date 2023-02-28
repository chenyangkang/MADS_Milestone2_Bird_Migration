
def get_variable_dict():
    sampling_variables=['duration_minutes',
                        'Traveling',
                        'Stationary',
                        'Area',
                        'obsvr_species_count']

    time_variables=['DOY',
                    'month',
                    'week',
                    'year',
                    'time_observation_started_minute_of_day']

    landscape_variables=[
                        'elevation_mean',
                        'slope_mean',
                        'eastness_mean',
                        'northness_mean',
                        'elevation_std',
                        'slope_std',
                        'eastness_std',
                        'northness_std']

    climate_variables=[
                    'prec',
                    'tmax',
                    'tmin',
                    'bio1',
                    'bio2',
                    'bio3',
                    'bio4',
                    'bio5',
                    'bio6',
                    'bio7',
                    'bio8',
                    'bio9',
                    'bio10',
                    'bio11',
                    'bio12',
                    'bio13',
                    'bio14',
                    'bio15',
                    'bio16',
                    'bio17',
                    'bio18',
                    'bio19',
                    ]

    land_use_variables=[
        'closed_shrublands',
        'closed_shrublands_ed',
        'closed_shrublands_lpi',
        'closed_shrublands_pd',
        'cropland_or_natural_vegetation_mosaics',
        'cropland_or_natural_vegetation_mosaics_ed',
        'cropland_or_natural_vegetation_mosaics_lpi',
        'cropland_or_natural_vegetation_mosaics_pd',
        'croplands',
        'croplands_ed',
        'croplands_lpi',
        'croplands_pd',
        'deciduous_broadleaf_forests',
        'deciduous_broadleaf_forests_ed',
        'deciduous_broadleaf_forests_lpi',
        'deciduous_broadleaf_forests_pd',
        'deciduous_needleleaf_forests',
        'deciduous_needleleaf_forests_ed',
        'deciduous_needleleaf_forests_lpi',
        'deciduous_needleleaf_forests_pd',
        'evergreen_broadleaf_forests',
        'evergreen_broadleaf_forests_ed',
        'evergreen_broadleaf_forests_lpi',
        'evergreen_broadleaf_forests_pd',
        'evergreen_needleleaf_forests',
        'evergreen_needleleaf_forests_ed',
        'evergreen_needleleaf_forests_lpi',
        'evergreen_needleleaf_forests_pd',
        'grasslands',
        'grasslands_ed',
        'grasslands_lpi',
        'grasslands_pd',
        'mixed_forests',
        'mixed_forests_ed',
        'mixed_forests_lpi',
        'mixed_forests_pd',
        'non_vegetated_lands',
        'non_vegetated_lands_ed',
        'non_vegetated_lands_lpi',
        'non_vegetated_lands_pd',
        'open_shrublands',
        'open_shrublands_ed',
        'open_shrublands_lpi',
        'open_shrublands_pd',
        'permanent_wetlands',
        'permanent_wetlands_ed',
        'permanent_wetlands_lpi',
        'permanent_wetlands_pd',
        'savannas',
        'savannas_ed',
        'savannas_lpi',
        'savannas_pd',
        'urban_and_built_up_lands',
        'urban_and_built_up_lands_ed',
        'urban_and_built_up_lands_lpi',
        'urban_and_built_up_lands_pd',
        'water_bodies',
        'water_bodies_ed',
        'water_bodies_lpi',
        'water_bodies_pd',
        'woody_savannas',
        'woody_savannas_ed',
        'woody_savannas_lpi',
        'woody_savannas_pd',
        'entropy',
    ]


    return {
        'sampling_variables':sampling_variables,
        'time_variables':time_variables,
        'landscape_variables':landscape_variables,
        'climate_variables':climate_variables,
        'land_use_variables':land_use_variables
    }