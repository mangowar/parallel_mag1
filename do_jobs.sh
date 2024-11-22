for i in $(ls ../OMP_jobs)
do
bsub <  $i
echo $i
done