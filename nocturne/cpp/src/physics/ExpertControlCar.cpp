#include "ExpertControlCar.h"
#include <iostream>

namespace physics {

ExpertControlCar::ExpertControlCar(float width,float length, b2World *world) : BaseCar(width,length)
{
    m_World = world!=nullptr?world:Getb2World();

    // create the body of the car...
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody; //b2_staticBody; 
    bodyDef.position.Set(0.f,0.f);
    m_Body = m_World->CreateBody(&bodyDef);
    b2PolygonShape shape; 
    shape.SetAsBox(width/2,length/2); 
    m_Body->CreateFixture(&shape,20.f);
   
}

ExpertControlCar::~ExpertControlCar()
{

}


void ExpertControlCar::Step(float dt)
{
//    std::cout << "ExpertControlCar::Step dt=" << dt << std::endl;
}

}   // namespace physics
