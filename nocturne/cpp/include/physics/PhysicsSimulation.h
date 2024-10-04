#pragma once

#include "BaseCar.h"
#include <list>

namespace physics {

class PhysicsSimulation
{
    public:
        PhysicsSimulation();
        virtual ~PhysicsSimulation();

        void Step(float dt);

        void AddCar(BaseCar *car);
        void RemoveCar(BaseCar *car);
        void DeleteScene();
        void DeleteAllBodies(b2World *b2world);

    protected:
//        std::list<Trajectory*> m_Trajectories;
        std::list<BaseCar*> m_Cars;
//        std::list<TrajectoryCar*> m_TrajectoryCars;
//        FreeCar* m_FreeCar;
};


}   // namespace physics
