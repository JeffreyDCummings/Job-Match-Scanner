# Job-Match-Scanner
Applied TF-IDF and keyword matching to scan key skills in job postings and rate/optimize resumes and cover letters.

This program reads in txt versions of job postings within the current directory (those named *job_post.txt), and scans them
for data science keywords and highly rated TF-IDF terms.  Then, these terms are output for each posting, and matches/scores are 
output for the input pdf resume.  Therefore, suggestions are made on key skills you should add to your resume for that given job,
if not learn.  These top rated terms for each posting will also help guide your focus in your cover letter written for that job.
