-- load data
logs = LOAD 'data/access_logs.csv' USING PigStorage(',') AS (AccessID:int, ByWho:int, WhatPage:int, TypeOfAccess:chararray, AccessTime:chararray);
pages = LOAD 'data/pages.csv' USING PigStorage(',') AS (PersonID:int, Name:chararray, Nationality:chararray, CountryCode:int, Hobby:chararray);

-- group by what page, and get a count
grouped_logs = GROUP logs BY WhatPage;
access_counts = FOREACH grouped_logs GENERATE group AS PageID, COUNT(logs) AS total_accesses;

-- order by most popular and keep 10
ordered_pages = ORDER access_counts BY total_accesses DESC;
top_10 = LIMIT ordered_pages 10;

-- join with pages to return name and nationality
joined_data = JOIN top_10 BY PageID, pages BY PersonID;
task_b = FOREACH joined_data GENERATE top_10::PageID, pages::Name, pages::Nationality;

DUMP task_b;