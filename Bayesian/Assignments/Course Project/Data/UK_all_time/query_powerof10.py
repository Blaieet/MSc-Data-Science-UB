from power_of_ten_query.po10 import client
import pandas as pd

po10_client = client.Client()
# ranking_Decathlon = po10_client.get_ranking(event="Dec", sex="M", age_group="ALL", alltime="y")
ranking_Decathlon = po10_client.get_ranking(event="HepW", sex="W", age_group="ALL", alltime="y")

ids = [r.athlete.id for r in ranking_Decathlon]
# ids = ids[98:len(ids)+1]
i = 0
for athlete in ids:
    print(i)
    i += 1
    print(athlete)
    athlete_pbs = pd.read_html('https://www.thepowerof10.info/athletes/profile.aspx?athleteid='+str(athlete))
    athlete_pbs = [tb for tb in athlete_pbs if tb.shape[1] > 2]
    athlete_pbs = [tb for tb in athlete_pbs if tb.iloc[0, 1] == 'PB'][0]
    athlete_pbs.columns = athlete_pbs.iloc[0,:]
    athlete_pbs.index = athlete_pbs.iloc[:,0]
    athlete_pbs.drop('Event', axis=0, inplace=True)
    athlete_pbs.drop('Event', axis=1, inplace=True)
    athlete_pbs.to_csv('power_of_ten_PBs/heptathlon/'+str(athlete)+'.csv')



