// Cpp file for Binary Classification Machine Learning
#include<bits/stdc++.h>
using namespace std;
bool custom_sort(double a, double b)

{
    double  a1=abs(a-0);
    double  b1=abs(b-0);
    return a1<b1;
}
int main()
{
/*Dataset feeding Phase*/
    double SL[] = {5.1,4.8,5.1,6.7,5.4,5.5,6.2,6.0,5.5,5.6,5.4,4.5,5.5,5.5,6.1,5.1,6.9,6.2,
                   5.9,4.9,5.3,5.0,6.4,5.2,4.9,5.6,5.1,4.9,5.5,5.2,4.4,6.0,4.8,5.7,6.1,
                   5.1,5.0,5.4,5.0,5.2,5.8,4.6,5.0,5.7,5.1,4.7,5.8,5.1};     //Sepal length data
    double SW[] = {3.5,3.0,3.8,3.0,3.4,2.4,2.2,3.4,2.4,3.0,3.9,2.3,3.5,4.2,2.9,2.5,3.1,2.9,
                   3.2,2.4,3.7,3.4,3.2,2.7,3.0,2.7,3.7,3.1,2.6,3.4,2.9,2.7,3.0,3.0,2.8,
                   3.8,2.3,3.0,3.0,3.5,2.7,3.2,3.2,2.8,3.3,3.2,2.6,3.5};     //Sepal Width
    double PL[] = {1.4,1.4,1.9,5.0,1.5,3.7,4.5,4.5,3.8,4.5,1.3,1.3,1.3,1.4,4.7,3.0,4.9,4.3,
                   4.8,3.3,1.5,1.6,4.5,3.9,1.4,4.2,1.5,1.5,4.4,1.4,1.4,5.1,1.4,4.2,4.0,
                   1.6,3.3,4.5,1.6,1.5,4.1,1.4,1.2,4.5,1.7,1.6,4.0,1.4};     //Petal Length
    double PW[] = {0.2,0.3,0.4,1.7,0.4,1.0,1.5,1.6,1.1,1.5,0.4,0.3,0.2,0.2,1.4,1.1,1.5,1.3,
                   1.8,1.0,0.2,0.4,1.5,1.4,0.2,1.3,0.4,0.1,1.2,0.2,0.2,1.6,0.1,1.2,1.3,
                   0.2,1.0,1.5,0.2,0.2,1.0,0.2,0.2,1.3,0.5,0.2,1.2,0.3};      //Petal Width
    double y[] = {0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,1,1,0,
                  1,1,0,0,1,0,0,1,0,0,1,0};                                   //data OUTPUT
    vector<double>error;                          // for storing the error values
    double err;                                   // for calculating error on each stage
    double b1,b2,b3,b4,b5,b6 = 0;                 //bias Declaration
    double SL1 = 0;
    double SL2 = 0;                               //weights Declaration
    double SL3 = 0;
    double SL4 = 0;
    double SL5 = 0;
    double W1,W2,W3,W4,W5 = 0;
    double SW1,SW2,SW3,SW4,SW5 = 0;
    double PL1,PL2,PL3,PL4,PL5 = 0;
    double PW1,PW2,PW3,PW4,PW5 = 0;
    double alpha = 0.01;                           // initializing our learning rate
    double  e = 2.71828;                               //used in sigmoid

/*Training Phase*/
    for (int i = 0; i < 192; i ++) {       //Since there are 48 values in our dataset and we want to run for 4 epochs so total for loop run 192 times
        int idx = i % 48;                 //for accessing index after every epoch
        double p1 = -(b1 + SL1* SL[idx]+ SW1* SW[idx]+ PL1* PL[idx]+ PW1* PW[idx]);  //input of node p1
        double p2 = -(b2 + SL2* SL[idx]+ SW2* SW[idx]+ PL2* PL[idx]+ PW2* PW[idx]);  //input of node p2
        double p3 = -(b3 + SL3* SL[idx]+ SW3* SW[idx]+ PL3* PL[idx]+ PW3* PW[idx]);  //input of node p3
        double p4 = -(b4 + SL4* SL[idx]+ SW4* SW[idx]+ PL4* PL[idx]+ PW4* PW[idx]);  //input of node p4
        double p5 = -(b5 + SL5* SL[idx]+ SW5* SW[idx]+ PL5* PL[idx]+ PW5* PW[idx]);  //input of node p5

        //making the prediction
        double pred1  = 1/(1+ pow(e,p1));    //Activating Node 1
        double pred2  = 1/(1+ pow(e,p2));    //Activating Node 2
        double pred3  = 1/(1+ pow(e,p3));    //Activating Node 3
        double pred4  = 1/(1+ pow(e,p4));    //Activating Node 4
        double pred5  = 1/(1+ pow(e,p5));    //Activating Node 5

        double node_last = -(b6 + W1* pred1 + W2* pred2 + W3* pred3 + W4* pred4 + W5*pred5);    //input to output node
        double output = 1/(1+ pow(e,node_last));   //Activating Output Node to give value in between 0 & 1
                                                       //calculating final prediction applying sigmoid
        err = y[idx]-output;             //calculating the error



        //  Phase- 2 -Backpropagation


        b1 = b1 - alpha * err*pred1 *(1-pred1)* 1.0;    //Updating Bias corresponding to Node 1
        b2 = b2 - alpha * err*pred2 *(1-pred2)* 1.0;    //Updating Bias corresponding to Node 2
        b3 = b3 - alpha * err*pred3 *(1-pred3)* 1.0;    //Updating Bias corresponding to Node 3
        b4 = b4 - alpha * err*pred4 *(1-pred4)* 1.0;    //Updating Bias corresponding to Node 4
        b5 = b5 - alpha * err*pred5 *(1-pred5)* 1.0;    //Updating Bias corresponding to Node 5
          b6 = b6 - alpha * err*output *(1-output)* 1.0;   //Updating Bias for Output Node
            SL1 = SL1 + alpha * err * pred1*(1-pred1) * SL[idx];
            SL2 = SL2 + alpha * err * pred2*(1-pred2) * SL[idx];    //Updating Weights by Gradient descent associated with Sepal Length INPUT
            SL3 = SL3 + alpha * err * pred3*(1-pred3) * SL[idx];
            SL4 = SL4 + alpha * err * pred4*(1-pred4) * SL[idx];
            SL5 = SL5 + alpha * err * pred5*(1-pred5) * SL[idx];

                             SW1 = SW1 + alpha * err * pred1*(1-pred1) * SW[idx];
                             SW2 = SW2 + alpha * err * pred2*(1-pred2) * SW[idx];
                             SW3 = SW3 + alpha * err * pred3*(1-pred3) * SW[idx];
                             SW4 = SW4 + alpha * err * pred4*(1-pred4) * SW[idx];
                             SW5 = SW5 + alpha * err * pred5*(1-pred5) * SW[idx];
                                   PL1 = PL1 + alpha * err * pred1*(1-pred1) * PL[idx];
                                   PL2 = PL2 + alpha * err * pred2*(1-pred2) * PL[idx];
                                   PL3 = PL3 + alpha * err * pred3*(1-pred3) * PL[idx];
                                   PL4 = PL4 + alpha * err * pred4*(1-pred4) * PL[idx];
                                   PL5 = PL5 + alpha * err * pred5*(1-pred5) * PL[idx];
                 PW1 = PW1 + alpha * err * pred1*(1-pred1) * PW[idx];
                 PW2 = PW2 + alpha * err * pred2*(1-pred2) * PW[idx];
                 PW3 = PW3 + alpha * err * pred3*(1-pred3) * PW[idx];
                 PW4 = PW4 + alpha * err * pred4*(1-pred4) * PW[idx];
                 PW5 = PW5 + alpha * err * pred5*(1-pred5) * PW[idx];
        W1 = W1 + alpha * err * output*(1-output) * pred1;
        W2 = W2 + alpha * err * output*(1-output) * pred2;
        W3 = W3 + alpha * err * output*(1-output) * pred3;
        W4 = W4 + alpha * err * output*(1-output) * pred4;
        W5 = W5 + alpha * err * output*(1-output) * pred5;
        cout<<"B1="<<b1<<" "<<"B2="<<b2<<" "<<"B3="<<b3<<" "<<"B4="<<b4<<" "<<"B5="<<b5<<" "<<"B6="<<b6
        <<"SL1="<<SL1<<" "<<"SL2="<<SL2<<" "<<"SL3="<<SL3<<" "<<"SL4="<<SL4<<" "
        <<"SL5="<<SL5<<" "<<"SW1="<<SW1<<" "<<"SW2="<<SW2<<" "<<"SW3="<<SW3<<" "<<"SW4="
        <<SW4<<" "<<"SW5="<<SW5<<" "<<"PL1="<<PL1<<" "<<"PL2="<<PL2<<" "<<"PL3="
        <<PL3<<" "<<"PL4="<<PL4<<" "<<"PL5="<<PL5<<" "<<"PW1="<<PW1<<" "
        <<"PW2="<<PW2<<" "<<"PW3="<<PW3<<" "<<"PW4="<<PW4<<" "<<"PW5="<<PW5
            <<" error="<<err<<endl;                     // printing values after every step
        error.push_back(err);
    }
    sort(error.begin(),error.end(),custom_sort);    //custom sort based on absolute error difference
    cout<<"Final Values are: "<<"B1="<<b1<<" "<<"B2="<<b2<<" "<<"B3="<<b3<<" "<<"B4="<<b4<<" "
    <<"B5="<<b5<<" "<<"B6="<<b6
    <<"SL1="<<SL1<<" "<<"SL2="<<SL2<<" "<<"SL3="<<SL3<<" "<<"SL4="<<SL4<<" "
    <<"SL5="<<SL5<<" "<<"SW1="<<SW1<<" "<<"SW2="<<SW2<<" "<<"SW3="<<SW3<<" "<<"SW4="
    <<SW4<<" "<<"SW5="<<SW5<<" "<<"PL1="<<PL1<<" "<<"PL2="<<PL2<<" "<<"PL3="
    <<PL3<<" "<<"PL4="<<PL4<<" "<<"PL5="<<PL5<<" "<<"PW1="<<PW1<<" "
    <<"PW2="<<PW2<<" "<<"PW3="<<PW3<<" "<<"PW4="<<PW4<<" "<<"PW5="<<PW5<<"error="<<error[0];

/*Testing Phase*/


    double test1,test2,test3,test4;
    cin>>test1>>test2>>test3>>test4;                                //Enter value of test data in order of SL,SW,PL,PW
       double test_out1 = -(b1 + SL1*test1 + SW1*test2 + PL1*test3 + PW1*test4);  //test data going through node 1
       double test_out2 = -(b2 + SL2*test1 + SW2*test2 + PL2*test3 + PW2*test4);  //test data going through node 2
       double test_out3 = -(b3 + SL3*test1 + SW3*test2 + PL3*test3 + PW3*test4);  //test data going through node 3
       double test_out4 = -(b4 + SL4*test1 + SW4*test2 + PL4*test3 + PW4*test4);  //test data going through node 4
       double test_out5 = -(b5 + SL5*test1 + SW5*test2 + PL5*test3 + PW5*test4);  //test data going through node 5
    double test_output1 = 1/1+(pow(e,test_out1)); //Activating node 1 containing train data
    double test_output2 = 1/1+(pow(e,test_out2));  //Activating node 2 containing train data
    double test_output3 = 1/1+(pow(e,test_out3));  //Activating node 3 containing train data
    double test_output4 = 1/1+(pow(e,test_out4));   //Activating node 4 containing train data
    double test_output5 = 1/1+(pow(e,test_out5));   //Activating node 5 containing train data
       double o1 = -(b6 + W1*test_output1 + W2*test_output2 + W3*test_output3 + W4*test_output4 + W5*test_output5);
       double final_output = 1/1+(pow(e,o1));       //final output
    cout<<"The value predicted by the model= "<<final_output<<endl;
    if(final_output>0.5) {
        final_output = 1;
    }
    else
    {
        final_output = 0;
    }

    cout<<"The class predicted by the model= "<<final_output;
}

