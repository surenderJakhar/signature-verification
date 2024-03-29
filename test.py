import preproc
import features
import os
import dataset

def testing(path):
    feature = dataset.getCSVFeatures(path)
    #dataset.makeCSV()
    if not(os.path.exists('data/TestFeatures')):
        os.mkdir('data/TestFeatures')
    with open('data/TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')


if __name__=="__main__":
    main()
