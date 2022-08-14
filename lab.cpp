#include <iostream>
using namespace std;

 enum lab {meow =3 };

lab foo()
{
    return meow;
}

int foo1()
{
    return meow;
}

int foo1(int d)
{
    return meow;
}
int main()
{

   
    cout<<foo1()<<endl;
    cout<<foo1(3)<<endl;

}

