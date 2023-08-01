import seaborn as sns
import folium
import geopandas as gpd
CHICAGO_COORD = [41.86364, -87.72645]

def create_top_ten_map(best_indexes = [], worst_indexes = []):
    """
    This function creates a folium GeoJson map with the boundaries of given Community Areas.
    """

    geo_json = gpd.read_file("data/census_tract.geojson")
    
    base_map = folium.Map(location=CHICAGO_COORD, tiles="cartodbpositron")
    folium.GeoJson(
        data=geo_json,
        popup=folium.GeoJsonPopup(fields=["geoid10","commarea_n","name10"]),
        style_function=lambda x: {"fillColor": "green", "color": "black", "weight": 1},
    ).add_to(base_map)

    geo_json["geoid10"] = geo_json["geoid10"].astype(float)

    for i in best_indexes:
        filtered = geo_json[geo_json["geoid10"] == i]
        folium.GeoJson(
            data=filtered,
            popup=folium.GeoJsonPopup(fields=["geoid10","commarea_n","name10"]),
            style_function=lambda x: {"fillColor": "blue", "color": "black", "weight": 3},
        ).add_to(base_map)

    for i in worst_indexes:
        filtered = geo_json[geo_json["geoid10"] == i]
        folium.GeoJson(
            data=filtered,
            popup=folium.GeoJsonPopup(fields=["geoid10","commarea_n","name10"]),
            style_function=lambda x: {"fillColor": "red", "color": "black", "weight": 2},
        ).add_to(base_map)

    return base_map

def descr_stat(dataframe, columns=[], group_by=[], sort=False, sort_by=[], as_index=True, agg_mode="count", plot=False, plot_map=False, ):
    
    grouped = dataframe[columns].groupby(group_by, as_index=as_index)
    
    if agg_mode=="count": 
        agg_mode_l = "Count" 
        grouped = grouped.count()
    if agg_mode=="avg" or agg_mode=="mean":
        agg_mode_l = "Average" 
        grouped = grouped.mean()
    if agg_mode=="sum":
        agg_mode_l = "Sum"
        grouped = grouped.sum()

    grouped = grouped.rename(columns={columns[-1]: agg_mode_l})

    if(sort==True):
        if (len(sort_by) == 1):
            grouped = grouped.sort_values(by=agg_mode_l)
        else:
            grouped = grouped.sort_values(by=sort_by)

    print("Rename done")
    if(plot==True):
        if(as_index==False):
            grouped = grouped.sort_values(by=group_by)[[agg_mode_l]]
            sns.lineplot(grouped)
        else:
            grouped = grouped
            sns.lineplot(grouped)

    print("Head: \n")
    print(grouped.head())
    print("Head-Indexes:")
    print(grouped.head().index)
    print("Tail: \n")
    print(grouped.tail())
    # if(as_index==True):
    print("Tail-Indexes:")
    print(grouped.tail().index)

    map = 0

    if (plot_map == True):
        map = create_top_ten_map(grouped.head().index, grouped.tail().index)

    return grouped, map

    