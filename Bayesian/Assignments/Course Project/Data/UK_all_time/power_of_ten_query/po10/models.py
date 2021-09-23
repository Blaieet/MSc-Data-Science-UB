from schematics import models
from schematics.types import StringType, IntType
from schematics.types.compound import ListType, ModelType





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
    