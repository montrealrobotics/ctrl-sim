#include "FreeCar.h"
#include "defines.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <string>

#include "Singletons.h"

namespace physics {

FreeCar::FreeCar(float width,float length, b2World *world) : BaseCar(width,length)
{
    m_World = world!=nullptr?world:Getb2World();

    // vehicle characteristics
    m_MaxSpeed =                FREECAR_MAXSPEED;
    m_MaxReverseSpeed =         FREECAR_MAXREVERSESPEED;
    m_MaxThrottleAccel =        FREECAR_MAXTHROTTLEACCEL;
    m_MaxThrottleReverseAccel = FREECAR_MAXTHROTTLEREVERSEACCEL;
    m_MaxBrakeAccel =           FREECAR_MAXBRAKEACCEL;
//    std::cout << "Ag length" << length << std::endl;
    m_MinTurnRadius =           length; // FREECAR_MINTURNRADIUS; // + (length - FREECAR_MINTURNRADIUS)*0.5;
    m_SideSpeedDamping =        FREECAR_SIDESPEEDDAMPING;
    m_AngularDamping =          FREECAR_ANGULARDAMPING; // angular damping in rad.s-2 to achieve dry friction

    // vehicle status
    m_ThrottleAccel = 0.f;
    m_BrakeAccel = 0.f;
    m_Steering = 0.f;
//    m_Speed = 0.f;
    
    // create the body of the car...
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody; // speed set by user, moved by solver
    bodyDef.position.Set(0.f,0.f);
    m_Body = m_World->CreateBody(&bodyDef);
    b2PolygonShape shape; 
    shape.SetAsBox(m_Width/2,m_Length/2); 
    m_Body->CreateFixture(&shape,20.f);

    // position the body
    // Respawn();
    m_Body->SetTransform(b2Vec2(0,0),0.f);
    m_Body->SetAngularVelocity(0.f);
    m_Body->SetLinearVelocity(b2Vec2(0,0));

    
}

FreeCar::~FreeCar()
{
    // TODO
}

// void FreeCar::Respawn()
// {
//     float x = m_SpawnDistance * (-1.f+2.f*(((float)rand()) / (float)RAND_MAX));
//     float y = m_SpawnDistance * (-1.f+2.f*(((float)rand()) / (float)RAND_MAX));
//     b2Vec2 position(x,y);
//     m_Body->SetTransform(position,0.f);
//     m_Body->SetAngularVelocity(0.f);
//     m_Body->SetLinearVelocity(b2Vec2(0,0));
// }

void FreeCar::Throttle(float value)
{
    if (value>0)
        m_ThrottleAccel = m_MaxThrottleAccel * value;
    else
        m_ThrottleAccel = m_MaxThrottleReverseAccel * value;
    m_BrakeAccel = 0.f;     // could be matter of design discussion
}

void FreeCar::Brake(float value)
{
    if (fabsf(value)<0.001) return;
    m_ThrottleAccel = 0;    // could be matter of design discussion
    m_BrakeAccel = m_MaxBrakeAccel * value;

}

void FreeCar::Turn(float value)
{
    m_Steering = value;
}

float DampenSpeed(float speed,float targetspeed,float damping,float dt)
{
    float speed_reduction = damping*dt;
    if (speed-targetspeed>speed_reduction)
        return speed-speed_reduction;
    if (speed-targetspeed<-speed_reduction)
        return speed+speed_reduction;
    return targetspeed;
}

void FreeCar::Step(float dt)
{
    // Compute accelerations
    float speedTarget = 0.f;
    float acceleration = 0.f;
    if (m_ThrottleAccel > 0.f)
    {
        // accel forward
        if (m_ThrottleAccel>m_BrakeAccel)
        {
            // accelerate
            speedTarget = m_MaxSpeed;
            acceleration = m_ThrottleAccel - m_BrakeAccel;
        }
        else
        {
            // brake
            speedTarget = 0.f;
            acceleration = m_BrakeAccel - m_ThrottleAccel;
        }
    }
    else
    {
        // accel backward
        if (m_ThrottleAccel<-m_BrakeAccel)
        {
            // accelerate
            speedTarget = m_MaxReverseSpeed;
            acceleration = -m_ThrottleAccel - m_BrakeAccel;
        }
        else
        {
            // brake
            speedTarget = 0.f;
            acceleration = m_BrakeAccel + m_ThrottleAccel;
        }
    }

    /*
    Free car update:
    - compute speed (decomposed in f/r where f = front speed, r=lateral(right) speed)
    - dampen lateral speed to 0
    - dampen front speed to speedtarget
    */ 

    // compute front and right unit vectors
    float Speed_angular = m_Body->GetAngularVelocity();
    float beta = std::atan(0.5 * std::tan(m_Steering));
    
    float c = cosf(m_Body->GetAngle() + beta);
    float s = sinf(m_Body->GetAngle() + beta);
    
    b2Vec2 forward(-s,c);   // Y axis=forward, X=right
    b2Vec2 right(c,s);
 
    // compute current speed
    b2Vec2 S = m_Body->GetLinearVelocity();
    float Speed_forward = b2Dot(S,forward);
    float Speed_right = b2Dot(S,right);
    
    // Compute Speed (throttle / brakes)
    float deltaV = acceleration * dt;
    if (Speed_forward<speedTarget)
        // accelerate forward 
        Speed_forward = fminf(Speed_forward+deltaV,speedTarget);   
    else
        // accelerate backward
        Speed_forward = fmaxf(Speed_forward-deltaV,speedTarget);


    // steering
    float steering_angular_speed = 0.f;
    if (fabs(m_Steering)>0.0000001)
    {
        float ray = 1.f/tanf(m_Steering) * m_MinTurnRadius/cosf(beta);
        steering_angular_speed = Speed_forward/ray;
    }

    // speeds damping
    Speed_right = DampenSpeed(Speed_right,0,m_SideSpeedDamping,dt);
    Speed_angular = DampenSpeed(Speed_angular,steering_angular_speed,m_AngularDamping,dt);

    b2Vec2 sR = right; sR *= Speed_right;
    b2Vec2 sF = forward; sF *= Speed_forward;
    b2Vec2 speed = sR+sF;
    m_Body->SetLinearVelocity(speed);
    m_Body->SetAngularVelocity(Speed_angular);

}



}   // namespace physics
