#pragma once

#include "BaseCar.h"

namespace physics {

class FreeCar : public BaseCar
{
    public:
        FreeCar(float width,float length, b2World *world = nullptr);
        virtual ~FreeCar();

        virtual void Step(float dt);

        void Throttle(float value);
        void Brake(float value);
        void Turn(float value);

    protected:
        b2World*    m_World;

        float m_MaxSpeed;
        float m_MaxReverseSpeed;
        float m_MaxThrottleAccel;
        float m_MaxThrottleReverseAccel;
        float m_MaxBrakeAccel;
        float m_MinTurnRadius;
        float m_SideSpeedDamping;
        float m_AngularDamping;

        float m_ThrottleAccel;
        float m_BrakeAccel;
        float m_Steering;
//        float m_Speed;

};

}   // namespace physics
