#pragma once


#include "BaseCar.h"
#include "Trajectory.h"

namespace physics {

class TrajectoryCar : public BaseCar
{
    public:
        TrajectoryCar(float width,float length, b2World *world, Trajectory *trajectory, float speed, float initialtrip=0.f);
        virtual ~TrajectoryCar();

        virtual void Step(float dt);

        const Trajectory* GetTrajectory() const {return m_Trajectory;}

    protected:
        Trajectory* m_Trajectory;
        float       m_Speed;
        float       m_Trip;
        b2World*    m_World;
};


}   // namespace physics
