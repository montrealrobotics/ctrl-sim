#include "Trajectory.h"

#include <iterator>
#include <math.h>


namespace physics {

Trajectory::Trajectory(std::list<b2Vec2>* waypoints)
{
    Init(waypoints);
}

Trajectory::Trajectory(std::list<float>* waypoints)
{
    std::list<b2Vec2> newlist;
    for (std::list<float>::iterator it = waypoints->begin(); it!=waypoints->end(); it++)
    {
        float x = *it++;
        float y = *it;
        newlist.push_back(b2Vec2(x,y));
    }
    Init(&newlist);
}

void Trajectory::Init(std::list<b2Vec2>* waypoints)
{
    if (waypoints->size()>=2)
    {
        std::list<b2Vec2>::iterator it=waypoints->begin();
        b2Vec2 P1=*it;
        float abscissa = 0.f;
        it++;
        for (; it!=waypoints->end(); it++)
        {
            b2Vec2 P2 = *it;
            Segment* s = new Segment;
            s->P1 = P1;
            s->P2 = P2;
            b2Vec2 D = P2 - P1;
            s->length = D.Normalize();
            s->N = D;
            s->abscissa = abscissa;
            m_Segments.push_back(s);
            abscissa += s->length;
            P1 = P2;
        }

    }
}

Trajectory::~Trajectory()
{
    // delete all segments in memory
    while (!m_Segments.empty())
    {
        delete m_Segments.back();
        m_Segments.pop_back();
    }
}

float Trajectory::GetLength()
{
    if (!m_Segments.empty())
    {
        Segment *s = m_Segments.back();
        return s->length + s->abscissa;
    }
    return 0.f;
}

// int Trajectory::GetNbSegments()
// {
//     return m_Segments.size();
// }

void Trajectory::GetPosition(float abscissa, b2Vec2 &position)
{
    Segment *s = GetSegment(abscissa);
    position = s->N;
    position *= (abscissa-s->abscissa);
    position += s->P1;
}

void Trajectory::GetDirection(float abscissa, b2Vec2 &direction)
{
    Segment *s = GetSegment(abscissa);
    direction = s->N;
}

float Trajectory::GetAngle(float abscissa)
{
    Segment *s = GetSegment(abscissa);
    if (!s) 
        return 0.f;
    float angle = acosf(s->N.y);
    if (s->N.x>0)
        angle = -angle;
    return angle;
}

Trajectory::Segment* Trajectory::GetSegment(float abscissa)
{
    for (std::list<Segment*>::iterator it = m_Segments.begin(); it!=m_Segments.end(); it++)
    {
        Segment *s = *it;
        if (s->abscissa + s->length > abscissa)
            return s;
    }
    return nullptr;
}

std::string Trajectory::Repr()
{
    std::string repr = "<car2dphysics.Trajectory object nbsegments="+std::to_string(m_Segments.size())+">";
    return repr;
}


}   // namespace physics
