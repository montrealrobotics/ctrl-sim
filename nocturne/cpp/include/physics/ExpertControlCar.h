#pragma once

#include "BaseCar.h"
#include "Singletons.h"

namespace physics {

class ExpertControlCar : public BaseCar
{
    public:
        ExpertControlCar(float width,float length, b2World *world=nullptr);
        virtual ~ExpertControlCar();

        virtual void Step(float dt);

    protected:
        b2World*    m_World;
};

}   // namespace physics
