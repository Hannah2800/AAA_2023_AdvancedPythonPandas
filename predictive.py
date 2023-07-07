def dayandseas_to_int(df):
    
    df['dayOfWeek'] = df['dayOfWeek'].replace(['Monday', 'Tuesday','Wednesday','Thursday',
                                                                  'Friday','Saturday','Sunday'],[0,1,2,3,4,5,6])
    df['season'] = df['season'].replace(['summer', 'winter','spring','autumn'],[0,1,2,3])

    return df



