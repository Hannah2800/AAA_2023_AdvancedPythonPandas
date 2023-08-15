import seaborn as sns
import folium
import geopandas as gpd
from h3 import h3
import json
import pandas as pd
import matplotlib.pyplot as plt

city_bounding = gpd.read_file('../data/boundaries_city.geojson')
city_bounding_json_string = city_bounding.to_json()
city_bounding_json = json.loads(city_bounding_json_string)
city_bounding_poly = city_bounding_json["features"][0]

# Create a GeoDataFrame from the MultiPolygon
gdf = gpd.GeoDataFrame.from_features([city_bounding_poly])

# Explode the MultiPolygon into individual polygons
exploded_city_bounding_poly = gdf.explode()

# Get both Polygons
cityBoundingPolygonSmall = {'type': 'Polygon', 'coordinates': city_bounding_poly["geometry"]["coordinates"][0]}
cityBoundingPolygonBig = {'type': 'Polygon', 'coordinates': city_bounding_poly["geometry"]["coordinates"][1]}

CHICAGO_COORD = [41.86364, -87.72645]

def create_top_ten_map(best_indexes = [], worst_indexes = []):
    """
    This function creates a folium GeoJson map with the boundaries of given Community Areas.
    """

    geo_json = gpd.read_file("../data/census_tract.geojson")
    
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

    # print("Rename done")
    if(plot==True):
        if(as_index==False):
            grouped = grouped.sort_values(by=group_by)[[agg_mode_l]]
            sns.lineplot(grouped)
            plt.grid()
        else:
            # grouped = grouped
            sns.lineplot(grouped)
            plt.grid()

    # print("Head: \n")
    # print(grouped.head())
    # print("Head-Indexes:", grouped.head().index)
    # print("Tail: \n")
    # print(grouped.tail())
    # print("Tail-Indexes:",grouped.tail().index)

    map = 0

    if (plot_map == True):
        map = create_top_ten_map(grouped.head().index, grouped.tail().index)

    return grouped, map



def visualize_hexagons(hexagons, color="red", folium_map=None):
    """
    hexagons is a list of hexcluster. Each hexcluster is a list of hexagons. 
    eg. [[hex1, hex2], [hex3, hex4]]
    """
    polylines = []
    lat = []
    lng = []
    for hex in hexagons:
        polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
        # flatten polygons into loops.
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        lat.extend(map(lambda v:v[0],polyline))
        lng.extend(map(lambda v:v[1],polyline))
        polylines.append(polyline)
    
    if folium_map is None:
        m = folium.Map(location=[sum(lat)/len(lat), sum(lng)/len(lng)], zoom_start=10, tiles='cartodbpositron')
    else:
        m = folium_map
    for polyline in polylines:
        my_PolyLine=folium.PolyLine(locations=polyline,weight=8,color=color)
        m.add_child(my_PolyLine)
    return m
    

def visualize_polygon(polyline, color):
    polyline.append(polyline[0])
    lat = [p[0] for p in polyline]
    lng = [p[1] for p in polyline]
    m = folium.Map(location=[sum(lat)/len(lat), sum(lng)/len(lng)], zoom_start=11, tiles='cartodbpositron')
    my_PolyLine=folium.PolyLine(locations=polyline,weight=8,color=color)
    m.add_child(my_PolyLine)
    return m

    
def timeperiod(df,period):
    df_tp = df.copy()
    df_tp['start_stamp'] = ((df_tp['start_time_hourly'] // period) * period).astype(int)
    
    return df_tp

#Col 1 as heaxagon or census tract, col2 as Price/trip minutes/etc.
def spat_and_temp(df,col1,period, col2):
    
    df_sp_tmp = timeperiod(df,period)
    
    df_sp_tmp['numOfTaxis_area'] = 0
    
    spa_temp = df_sp_tmp.groupby(['start_stamp','date_start',col1])['numOfTaxis_area'].count()
    price_sum = df_sp_tmp.groupby(['start_stamp','date_start',col1])[col2].sum()
    # price_avg = df_sp_tmp.groupby(['start_stamp','date_start',col1])[col2].mean()

    col_sum_for_rename = col2 + " Sum"
    # col_average_for_rename = col2 + " Average"
    
    subset = df_sp_tmp[['start_time_hourly','dayOfWeek','isHoliday','start_time_day','start_time_week',  
                 'start_time_month', "date_start",'start_stamp',col1]]
   
    spa_temp = pd.merge(spa_temp, subset, on=["date_start",'start_stamp',col1], how="left")
    spa_temp = pd.merge(spa_temp, price_sum, on=["date_start",'start_stamp',col1], how="left")
    spa_temp.rename(columns= {col2: col_sum_for_rename}, inplace=True)
    # spa_temp = pd.merge(spa_temp, price_avg, on=["date_start",'start_stamp',col1], how="left")
    # spa_temp.rename(columns= {col2: col_average_for_rename}, inplace=True)

    spa_temp.drop_duplicates(subset=["date_start",'start_stamp',col1], inplace = True)
    spa_temp.reset_index(drop=True, inplace=True)
    
    return spa_temp


def print_vals(dataframe):
    if(type(dataframe) is tuple):
        dataframe = dataframe[0]
        
    for i in range(len(dataframe.head())):
        print("Lowest values in:", dataframe.head().index[i], "with: ", round(dataframe.head().values[i][0], 2)) 
    for i in range(len(dataframe.tail())):
        print("Highest values in:", dataframe.tail().index[i], "with: ", round(dataframe.tail().values[i][0], 2))