#include "PhysicsSimulation.h"
#include "Singletons.h"
#include <iostream>
namespace physics {

PhysicsSimulation::PhysicsSimulation()
{

}

PhysicsSimulation::~PhysicsSimulation()
{

}

void PhysicsSimulation::Step(float dt)
{
    for (std::list<BaseCar*>::iterator it = m_Cars.begin(); it!=m_Cars.end(); it++)
        (*it)->Step(dt);

    // update world
    int velocityIterations = 8;
    int positionIterations = 3;
    Getb2World()->Step(dt,velocityIterations,positionIterations);
}

void PhysicsSimulation::AddCar(BaseCar *car)
{
    m_Cars.push_back(car);
}

void PhysicsSimulation::RemoveCar(BaseCar *car)
{
    m_Cars.remove(car);
}

void PhysicsSimulation::DeleteScene()
{
    for (auto it = m_Cars.begin(); it != m_Cars.end(); ++it)
    {
        delete *it;  // Deallocate memory for the car
    }
    m_Cars.clear();  // Clear the list of pointers
    DeleteAllBodies(Getb2World());
}

void PhysicsSimulation::DeleteAllBodies(b2World* world) {
    b2Body* body = world->GetBodyList();
    while (body != nullptr) {
        b2Body* nextBody = body->GetNext();
        world->DestroyBody(body);
        body = nextBody;
    }
}

}   // namespace physics
