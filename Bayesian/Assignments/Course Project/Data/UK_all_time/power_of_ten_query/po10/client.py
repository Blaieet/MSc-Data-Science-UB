from schematics import models
from schematics.types import StringType, IntType
from schematics.types.compound import ListType, ModelType

import requests
from bs4 import BeautifulSoup

class Athlete(models.Model):
    id = IntType()
    name = StringType()
    coach = StringType()
    clubs = StringType(serialized_name='Club:')
    gender = StringType(serialized_name='Gender:')
    age_group = StringType(serialized_name='Age Group:')
    county = StringType(serialized_name='County:')
    region = StringType(serialized_name='Region:')
    nation = StringType(serialized_name='Nation:')
    date_of_birth = StringType(serialized_name='Date of Birth:')
    
    def __repr__(self):
        return u"< Athlete: %s >" % self.name


class Ranking(models.Model):
    rank = IntType()
    time = StringType() # needs to change
    athlete = ModelType(Athlete)
    venue = StringType()
    date = StringType() # Again needs to change
    event = StringType()
    year = StringType()
    age_group = StringType()
    
    def __repr__(self):
        return '< Ranking: {0} {1} {2} {3} {4}>'.format(self.rank, self.time, self.athlete.name, self.event, self.year)
    

class Client(object):
    
    def get_athlete(self, id):
        r = requests.get("http://www.thepowerof10.info/athletes/profile.aspx", params={"athleteid": id})
        
        if r.status_code != 200:
            raise AttributeError("Unable to find athlete with id %s." % id)
        
        soup = BeautifulSoup(r.content, features="html.parser")
        
        a = Athlete({"id": id})
        
        name = soup.find_all(class_="athleteprofilesubheader")[0].h2.string.strip().encode("utf-8")
        a.name = name
        
        info = soup.find(id="ctl00_cphBody_pnlAthleteDetails").find_all('table')[2]    
        
        extra_details = {row.find_all("td")[0].string: row.find_all("td")[1].string for row in info.find_all("tr")}
        
        a.import_data(extra_details)
           
        try: 
            coach = soup.find(id="ctl00_cphBody_pnlAthleteDetails").find_all('table')[3].find("a").string.encode("utf-8")
            coach_url = soup.find(id="ctl00_cphBody_pnlAthleteDetails").find_all('table')[3].find("a").get('href')
            
            a.coach = coach
        except:
            pass
    
        return a
    
    
    def get_ranking(self, event="10K", sex="M", year="2014", age_group="ALL", alltime="n"):
        
        if alltime == 'y':
            params={"event": event, "agegroup": age_group, "sex": sex, "alltime": alltime}
            year = 'All_Time'
        else:
            params={"event": event, "agegroup": age_group, "sex": sex, "year": year}

        r = requests.get("http://www.thepowerof10.info/rankings/rankinglist.aspx", params=params)
        
        soup = BeautifulSoup(r.content, features="html.parser")
        rankings_table = soup.find(id="cphBody_lblCachedRankingList").find_all('table')[0]
        ranking_rows = [row for row in rankings_table.find_all("tr") if 'class' in row.attrs]
        ranking_rows = [row for row in ranking_rows if row["class"][0] not in ["rankinglisttitle", "rankinglistheadings", "rankinglistsubheader"]]
        rankings = []
        for row in ranking_rows:
            if row.find_all("td")[0].string is None:
                continue
            r = Ranking({"athlete": Athlete(), "event": event, "year": year, "age_group": age_group})
            r.rank = int(row.find_all("td")[0].string)
            r.time = row.find_all("td")[1].string
            r.athlete.name = row.find_all("td")[6].text
            r.athlete.id = int(row.find_all("td")[6].a["href"].split("=")[1])
            r.venue = row.find_all("td")[11].string
            r.date = row.find_all("td")[12].string
            
            rankings.append(r)
            
        return rankings
            
