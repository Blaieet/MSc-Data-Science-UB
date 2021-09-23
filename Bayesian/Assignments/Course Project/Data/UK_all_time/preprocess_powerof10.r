library(lubridate)
library(factoextra)

setwd("~/Documents/Master_DataScience/BayesianStatistics/Bayesian/Course Project/Data/UK_all_time/power_of_ten_PBs/")

# Necessary info to process decathlon and heptathlon differently
events_info <- list(decathlon=list(event_nm='decathlon', event_nm_short='Dec',events=c('100','LJ','SP7.26K','HJ','400','110H','DT2K','PV','JT800','1500'), distance='1500'),
                   heptathlon=list(event_nm='heptathlon', event_nm_short='HepW',events=c('100HW','HJ','SP4K','200','LJ','JT600','800'), distance='800'))

for (event in events_info){
  year_performance <- NULL
  
  # Process the files with the info for each of the athletes
  athletes_files <- list.files(paste0(event[['event_nm']],'/'))
  
  for (f in athletes_files){
    
    dt <- read.csv(paste0(event[['event_nm']],'/',f), header=TRUE) # Load dataset
    dt <- dt[!duplicated(dt[,1]), ] # Remove duplicated rows
    row.names(dt) <- dt[,1] # Set row names
    dt <- dt[ ,colnames(dt) != 'Event'] # Remove 'Event' columns (they are headers)
    colnames(dt) <- gsub('^X|\\.0$','',colnames(dt)) # Format years adequately
    
    # Process only if we have info for all events in the combined events
    if (sum(event[['events']] %in% row.names(dt)) == length(event[['events']])){
      
      # Keep only data on events in the combined events
      # and athletes that have data on a combined event
      dt <- dt[event[['events']],!is.na(dt[event[['event_nm_short']],])]
      dt <- apply(dt, MARGIN=2, function(x) gsub('i|w','',x)) # Remove indoor/wind indicators
      dt[dt=='DNF'|dt=='DNS'|dt=='NH'|dt=='NM'|dt==''] <- NA # Set missing performances to NA 
      
      # Remove years in which not all the events in the combined events have been completed
      dt <- dt[,colSums(is.na(dt)) == 0, drop=FALSE] 
      
      if (ncol(dt) > 1){
        # If we have two SB (wind/not-wind, indoor/outdoor...) keep the best one (longest distance, smallest time)
        dt['LJ',] <- sapply(dt['LJ',], function(x) max(as.numeric(strsplit(x, split='/')[[1]])))
        dt[row.names(dt) != event[['distance']], ] <- apply(dt[row.names(dt) != event[['distance']], ], MARGIN=c(1,2), function(x) min(as.numeric(strsplit(x, split='/')[[1]])))
        
        # 800m and 1500m performances are not in seconds. Transform them to seconds.
        dt[event[['distance']], ] <- period_to_seconds(ms(dt[event[['distance']],]))
        
        # Set adequate formats
        dt <- apply(dt, MARGIN=c(1,2), as.numeric)
        dt <- t(dt)
        dt <- as.data.frame(dt)
        
        # Merge to the inclusive dataframe
        dt$Year <- row.names(dt)
        dt$Athlete <- gsub('\\.csv','',f)
        year_performance <- rbind(year_performance, dt)
      }
    }
  }
  
  # Categorical variables to factors
  year_performance$Athlete <- factor(year_performance$Athlete)
  year_performance$Year <- factor(year_performance$Year)
  
  # Save results  
  saveRDS(year_performance, file=paste0('../../ProcessedDatasets/UK_all_time_',event[['event_nm']],'.RDS'))

}











